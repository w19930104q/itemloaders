"""
Asynchronous Item Loader

This module provides :class:`AsyncItemLoader`, an asynchronous variant of
:class:`~itemloaders.ItemLoader` whose value-processing methods are
coroutines, so that input, output and ad-hoc processors can be asynchronous
(e.g. to look values up in a remote dictionary service) without ever storing
coroutine objects as collected values.

Processors may freely mix regular functions, ``async`` functions and
callables returning awaitables; awaitable results are awaited in processor
declaration order and input value order. Use
:class:`~itemloaders.processors.AsyncCompose` and
:class:`~itemloaders.processors.AsyncMapCompose` as the asynchronous
counterparts of :class:`~itemloaders.processors.Compose` and
:class:`~itemloaders.processors.MapCompose`.

:class:`AsyncItemLoader` never starts an event loop itself: its methods are
coroutines that the caller must await within their own asynchronous
application.

Example:

>>> import asyncio
>>> from itemloaders import AsyncItemLoader
>>> from itemloaders.processors import AsyncMapCompose, TakeFirst
>>>
>>> async def lookup(value):
...     # e.g. an asynchronous dictionary lookup
...     return value.strip().title()
...
>>> class ProductLoader(AsyncItemLoader):
...     name_in = AsyncMapCompose(lookup)
...     name_out = TakeFirst()
...
>>> async def main():
...     loader = ProductLoader()
...     await loader.add_value('name', '  color tv ')
...     return await loader.load_item()
...
>>> asyncio.run(main())
{'name': 'Color Tv'}
"""

# AsyncItemLoader deliberately turns the processor-running methods of
# ItemLoader into coroutines; the signature change is the point of the class.
# pylint: disable=invalid-overridden-method

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten

from itemloaders import ItemLoader
from itemloaders.common import wrap_loader_context
from itemloaders.processors import _maybe_await
from itemloaders.utils import arg_to_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from re import Pattern


class AsyncItemLoader(ItemLoader):
    """
    Asynchronous variant of :class:`ItemLoader`.

    It accepts the same ``__init__`` arguments, attributes and Loader context
    as :class:`ItemLoader`, and nested loaders created with
    :meth:`~ItemLoader.nested_xpath` and :meth:`~ItemLoader.nested_css` share
    the collected values and inherit the context of their parent, exactly as
    with :class:`ItemLoader`.

    The difference is that every method that runs processors is a coroutine
    and must be awaited: :meth:`get_value`, :meth:`add_value`,
    :meth:`replace_value`, :meth:`load_item`, :meth:`get_output_value` and the
    selector-based methods (:meth:`add_xpath`, :meth:`replace_xpath`,
    :meth:`get_xpath`, :meth:`add_css`, :meth:`replace_css`, :meth:`get_css`,
    :meth:`add_jmes`, :meth:`replace_jmes` and :meth:`get_jmes`).

    Input processors, output processors and ad-hoc processors may be regular
    functions, ``async`` functions or callables returning awaitables, mixed
    freely; awaitable results are awaited in processor declaration order and
    input value order, so coroutine objects never end up stored as collected
    values.

    This class never starts an event loop: awaiting the coroutines is up to
    the calling asynchronous application.
    """

    async def add_value(  # type: ignore[override]
        self,
        field_name: str | None,
        value: Any,
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Process and then add the given ``value`` for the given field.

        Asynchronous variant of :meth:`ItemLoader.add_value`: it is a
        coroutine and the given ``processors`` may be asynchronous.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        value = await self.get_value(value, *processors, re=re, **kw)
        if value is None:
            return self
        if not field_name:
            for k, v in value.items():
                await self._add_value(k, v)
        else:
            await self._add_value(field_name, value)
        return self

    async def replace_value(  # type: ignore[override]
        self,
        field_name: str | None,
        value: Any,
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_value` but replaces the collected data with the
        new value instead of adding it.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        value = await self.get_value(value, *processors, re=re, **kw)
        if value is None:
            return self
        if not field_name:
            for k, v in value.items():
                await self._replace_value(k, v)
        else:
            await self._replace_value(field_name, value)
        return self

    async def _add_value(  # type: ignore[override]
        self, field_name: str, value: Any
    ) -> None:
        value = arg_to_iter(value)
        processed_value = await self._process_input_value(field_name, value)
        if processed_value:
            self._values.setdefault(field_name, [])
            self._values[field_name] += arg_to_iter(processed_value)

    async def _replace_value(  # type: ignore[override]
        self, field_name: str, value: Any
    ) -> None:
        self._values.pop(field_name, None)
        await self._add_value(field_name, value)

    async def get_value(
        self,
        value: Any,
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Process the given ``value`` by the given ``processors`` and keyword
        arguments.

        Asynchronous variant of :meth:`ItemLoader.get_value`: it is a
        coroutine and each processor may be a regular function, an ``async``
        function or a callable returning an awaitable. Processors are awaited
        in declaration order.
        """
        if re:
            value = arg_to_iter(value)
            value = flatten(extract_regex(re, x) for x in value)

        for proc in processors:
            if value is None:
                break
            _proc = proc
            proc = wrap_loader_context(proc, self.context)  # noqa: PLW2901
            try:
                value = await _maybe_await(proc(value))
            except Exception as e:
                raise ValueError(
                    f"Error with processor {_proc.__class__.__name__} "
                    f"value={value!r} error='{type(e).__name__}: {e!s}'"
                ) from e
        return value

    async def load_item(self) -> Any:
        """
        Populate the item with the data collected so far, and return it. The
        data collected is first passed through the :ref:`output processors
        <processors>` to get the final value to assign to each item field.
        """
        adapter = ItemAdapter(self.item)
        for field_name in tuple(self._values):
            value = await self.get_output_value(field_name)
            if value is not None:
                adapter[field_name] = value

        return adapter.item

    async def get_output_value(self, field_name: str) -> Any:
        """
        Return the collected values parsed using the output processor, for the
        given field. This method doesn't populate or modify the item at all.
        """
        proc = self.get_output_processor(field_name)
        proc = wrap_loader_context(proc, self.context)
        value = self._values.get(field_name, [])
        try:
            return await _maybe_await(proc(value))
        except Exception as e:
            raise ValueError(
                f"Error with output processor: field={field_name!r} "
                f"value={value!r} error='{type(e).__name__}: {e!s}'"
            ) from e

    async def _process_input_value(self, field_name: str, value: Any) -> Any:
        proc = self.get_input_processor(field_name)
        _proc = proc
        proc = wrap_loader_context(proc, self.context)
        try:
            return await _maybe_await(proc(value))
        except Exception as e:
            raise ValueError(
                f"Error with input processor {_proc.__class__.__name__}:"
                f" field={field_name!r} value={value!r} error='{type(e).__name__}: {e!s}'"
            ) from e

    async def add_xpath(  # type: ignore[override]
        self,
        field_name: str | None,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Asynchronous variant of :meth:`ItemLoader.add_xpath`.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_xpathvalues(xpath, field_name, **kw)
        return await self.add_value(field_name, values, *processors, re=re, **kw)

    async def replace_xpath(  # type: ignore[override]
        self,
        field_name: str | None,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_xpath` but replaces collected data instead of
        adding it.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_xpathvalues(xpath, field_name, **kw)
        return await self.replace_value(field_name, values, *processors, re=re, **kw)

    async def get_xpath(
        self,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Asynchronous variant of :meth:`ItemLoader.get_xpath`.
        """
        values = self._get_xpathvalues(xpath, **kw)
        return await self.get_value(values, *processors, re=re, **kw)

    async def add_css(  # type: ignore[override]
        self,
        field_name: str | None,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Asynchronous variant of :meth:`ItemLoader.add_css`.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_cssvalues(css, field_name)
        return await self.add_value(field_name, values, *processors, re=re, **kw)

    async def replace_css(  # type: ignore[override]
        self,
        field_name: str | None,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_css` but replaces collected data instead of
        adding it.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_cssvalues(css, field_name)
        return await self.replace_value(field_name, values, *processors, re=re, **kw)

    async def get_css(
        self,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Asynchronous variant of :meth:`ItemLoader.get_css`.
        """
        values = self._get_cssvalues(css)
        return await self.get_value(values, *processors, re=re, **kw)

    async def add_jmes(  # type: ignore[override]
        self,
        field_name: str | None,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Asynchronous variant of :meth:`ItemLoader.add_jmes`.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_jmesvalues(jmes, field_name)
        return await self.add_value(field_name, values, *processors, re=re, **kw)

    async def replace_jmes(  # type: ignore[override]
        self,
        field_name: str | None,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_jmes` but replaces collected data instead of
        adding it.

        :returns: The current AsyncItemLoader instance for method chaining.
        :rtype: AsyncItemLoader
        """
        values = self._get_jmesvalues(jmes, field_name)
        return await self.replace_value(field_name, values, *processors, re=re, **kw)

    async def get_jmes(
        self,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Asynchronous variant of :meth:`ItemLoader.get_jmes`.
        """
        values = self._get_jmesvalues(jmes)
        return await self.get_value(values, *processors, re=re, **kw)

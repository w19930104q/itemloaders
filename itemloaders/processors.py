"""
This module provides some commonly used processors for Item Loaders.

See documentation in :ref:`declaring-loaders`.
"""

from __future__ import annotations

import inspect
from collections import ChainMap
from typing import TYPE_CHECKING, Any

from itemloaders.common import wrap_loader_context
from itemloaders.utils import arg_to_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, MutableMapping


async def _maybe_await(value: Any) -> Any:
    """Return *value*, awaiting it first if it is an awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


class MapCompose:
    """
    A processor which is constructed from the composition of the given
    functions, similar to the :class:`Compose` processor. The difference with
    this processor is the way internal results are passed among functions,
    which is as follows:

    The input value of this processor is *iterated* and the first function is
    applied to each element. The results of these function calls (one for each element)
    are concatenated to construct a new iterable, which is then used to apply the
    second function, and so on, until the last function is applied to each
    value of the list of values collected so far. The output values of the last
    function are concatenated together to produce the output of this processor.

    Each particular function can return a value or a list of values, which is
    flattened with the list of values returned by the same function applied to
    the other input values. The functions can also return ``None`` in which
    case the output of that function is ignored for further processing over the
    chain.

    This processor provides a convenient way to compose functions that only
    work with single values (instead of iterables). For this reason the
    :class:`MapCompose` processor is typically used as input processor, since
    data is often extracted using the
    :meth:`~parsel.selector.Selector.extract` method of `parsel selectors`_,
    which returns a list of strings.

    The example below should clarify how it works:

    >>> def filter_world(x):
    ...     return None if x == 'world' else x
    ...
    >>> from itemloaders.processors import MapCompose
    >>> proc = MapCompose(filter_world, str.upper)
    >>> proc(['hello', 'world', 'this', 'is', 'something'])
    ['HELLO', 'THIS', 'IS', 'SOMETHING']

    As with the Compose processor, functions can receive Loader contexts, and
    ``__init__`` method keyword arguments are used as default context values.
    See :class:`Compose` processor for more info.

    .. _`parsel selectors`: https://parsel.readthedocs.io/en/latest/parsel.html#parsel.selector.Selector.extract
    """

    def __init__(self, *functions: Callable[..., Any], **default_loader_context: Any):
        self.functions = functions
        self.default_loader_context = default_loader_context

    def __call__(
        self, value: Any, loader_context: MutableMapping[str, Any] | None = None
    ) -> Iterable[Any]:
        values = arg_to_iter(value)
        context: MutableMapping[str, Any]
        if loader_context:
            context = ChainMap(loader_context, self.default_loader_context)
        else:
            context = self.default_loader_context
        wrapped_funcs = [wrap_loader_context(f, context) for f in self.functions]
        for func in wrapped_funcs:
            next_values: list[Any] = []
            for v in values:
                try:
                    next_values += arg_to_iter(func(v))
                except Exception as e:
                    raise ValueError(
                        f"Error in MapCompose with "
                        f"{func!s} value={value!r} error='{type(e).__name__}: {e!s}'"
                    ) from e
            values = next_values
        return values


class AsyncMapCompose(MapCompose):
    """
    Asynchronous variant of :class:`MapCompose`, for use with
    :class:`~itemloaders.AsyncItemLoader`.

    It works exactly like :class:`MapCompose` — the input value is iterated,
    each function is applied to every element in order, ``None`` results are
    filtered out and list results are flattened into the chain — but each
    function may be a regular function, an asynchronous function or a callable
    returning an awaitable, and the three kinds can be freely mixed. Awaitable
    results are awaited in function declaration order and input value order.

    The example below should clarify how it works:

    >>> import asyncio
    >>> def filter_world(x):
    ...     return None if x == 'world' else x
    ...
    >>> async def to_upper(x):
    ...     return x.upper()
    ...
    >>> from itemloaders.processors import AsyncMapCompose
    >>> proc = AsyncMapCompose(filter_world, to_upper)
    >>> asyncio.run(proc(['hello', 'world', 'this']))
    ['HELLO', 'THIS']

    As with :class:`MapCompose`, functions can receive Loader contexts, and
    ``__init__`` method keyword arguments are used as default context values.
    See :class:`Compose` processor for more info.
    """

    async def __call__(  # type: ignore[override]  # pylint: disable=invalid-overridden-method
        self, value: Any, loader_context: MutableMapping[str, Any] | None = None
    ) -> Iterable[Any]:
        values = arg_to_iter(value)
        context: MutableMapping[str, Any]
        if loader_context:
            context = ChainMap(loader_context, self.default_loader_context)
        else:
            context = self.default_loader_context
        wrapped_funcs = [wrap_loader_context(f, context) for f in self.functions]
        for func in wrapped_funcs:
            next_values: list[Any] = []
            for v in values:
                try:
                    next_values += arg_to_iter(await _maybe_await(func(v)))
                except Exception as e:
                    raise ValueError(
                        f"Error in AsyncMapCompose with "
                        f"{func!s} value={value!r} error='{type(e).__name__}: {e!s}'"
                    ) from e
            values = next_values
        return values


class Compose:
    """
    A processor which is constructed from the composition of the given
    functions. This means that each input value of this processor is passed to
    the first function, and the result of that function is passed to the second
    function, and so on, until the last function returns the output value of
    this processor.

    By default, stop process on ``None`` value. This behaviour can be changed by
    passing keyword argument ``stop_on_none=False``.

    Example:

    >>> from itemloaders.processors import Compose
    >>> proc = Compose(lambda v: v[0], str.upper)
    >>> proc(['hello', 'world'])
    'HELLO'

    Each function can optionally receive a ``loader_context`` parameter. For
    those which do, this processor will pass the currently active :ref:`Loader
    context <loaders-context>` through that parameter.

    The keyword arguments passed in the ``__init__`` method are used as the default
    Loader context values passed to each function call. However, the final
    Loader context values passed to functions are overridden with the currently
    active Loader context accessible through the :attr:`ItemLoader.context
    <itemloaders.ItemLoader.context>` attribute.
    """

    def __init__(self, *functions: Callable[..., Any], **default_loader_context: Any):
        self.functions = functions
        self.stop_on_none = default_loader_context.get("stop_on_none", True)
        self.default_loader_context = default_loader_context

    def __call__(
        self, value: Any, loader_context: MutableMapping[str, Any] | None = None
    ) -> Any:
        context: MutableMapping[str, Any]
        if loader_context:
            context = ChainMap(loader_context, self.default_loader_context)
        else:
            context = self.default_loader_context
        wrapped_funcs = [wrap_loader_context(f, context) for f in self.functions]
        for func in wrapped_funcs:
            if value is None and self.stop_on_none:
                break
            try:
                value = func(value)
            except Exception as e:
                raise ValueError(
                    f"Error in Compose with "
                    f"{func!s} value={value!r} error='{type(e).__name__}: {e!s}'"
                ) from e
        return value


class AsyncCompose(Compose):
    """
    Asynchronous variant of :class:`Compose`, for use with
    :class:`~itemloaders.AsyncItemLoader`.

    It works exactly like :class:`Compose` — each function receives the output
    of the previous one, and processing stops on ``None`` unless
    ``stop_on_none=False`` is passed — but each function may be a regular
    function, an asynchronous function or a callable returning an awaitable,
    and the three kinds can be freely mixed. Awaitable results are awaited in
    function declaration order.

    Example:

    >>> import asyncio
    >>> async def to_upper(v):
    ...     return v.upper()
    ...
    >>> from itemloaders.processors import AsyncCompose
    >>> proc = AsyncCompose(lambda v: v[0], to_upper)
    >>> asyncio.run(proc(['hello', 'world']))
    'HELLO'

    Each function can optionally receive a ``loader_context`` parameter, and
    the keyword arguments passed in the ``__init__`` method are used as the
    default Loader context values, exactly as with :class:`Compose`.
    """

    async def __call__(  # pylint: disable=invalid-overridden-method
        self, value: Any, loader_context: MutableMapping[str, Any] | None = None
    ) -> Any:
        context: MutableMapping[str, Any]
        if loader_context:
            context = ChainMap(loader_context, self.default_loader_context)
        else:
            context = self.default_loader_context
        wrapped_funcs = [wrap_loader_context(f, context) for f in self.functions]
        for func in wrapped_funcs:
            if value is None and self.stop_on_none:
                break
            try:
                value = await _maybe_await(func(value))
            except Exception as e:
                raise ValueError(
                    f"Error in AsyncCompose with "
                    f"{func!s} value={value!r} error='{type(e).__name__}: {e!s}'"
                ) from e
        return value


class TakeFirst:
    """
    Returns the first non-null/non-empty value from the values received,
    so it's typically used as an output processor to single-valued fields.
    It doesn't receive any ``__init__`` method arguments, nor does it accept Loader contexts.

    Example:

    >>> from itemloaders.processors import TakeFirst
    >>> proc = TakeFirst()
    >>> proc(['', 'one', 'two', 'three'])
    'one'
    """

    def __call__(self, values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return None


class Identity:
    """
    The simplest processor, which doesn't do anything. It returns the original
    values unchanged. It doesn't receive any ``__init__`` method arguments, nor does it
    accept Loader contexts.

    Example:

    >>> from itemloaders.processors import Identity
    >>> proc = Identity()
    >>> proc(['one', 'two', 'three'])
    ['one', 'two', 'three']
    """

    def __call__(self, values: Any) -> Any:
        return values


class SelectJmes:
    """
    Query the data structure for a *JMESPath* expression (given at instantiation), and return the answer.

    *Requires*: `JMESPath <https://github.com/jmespath/jmespath>`__

    .. note::

        ``SelectJmes`` accepts only one input element at a time.

    Example:

    >>> from itemloaders.processors import SelectJmes, Compose, MapCompose
    >>> proc = SelectJmes("foo") #for direct use on lists and dictionaries
    >>> proc({'foo': 'bar'})
    'bar'
    >>> proc({'foo': {'bar': 'baz'}})
    {'bar': 'baz'}

    Working with JSON:

    >>> import json
    >>> proc_single_json_str = Compose(json.loads, SelectJmes("foo"))
    >>> proc_single_json_str('{"foo": "bar"}')
    'bar'
    >>> proc_json_list = Compose(json.loads, MapCompose(SelectJmes('foo')))
    >>> proc_json_list('[{"foo":"bar"}, {"baz":"tar"}]')
    ['bar']
    """

    def __init__(self, json_path: str):
        self.json_path: str = json_path
        import jmespath.parser  # noqa: PLC0415

        self.compiled_path: jmespath.parser.ParsedResult = jmespath.compile(
            self.json_path
        )

    def __call__(self, value: Any) -> Any:
        """Query value for the JMESPath query and return answer
        :param value: a data structure (dict, list) to extract from
        :return: Element extracted according to JMESPath query
        """
        return self.compiled_path.search(value)


class Join:
    """
    Returns the values joined with the separator given in the ``__init__`` method, which
    defaults to ``' '``. It doesn't accept Loader contexts.

    When using the default separator, this processor is equivalent to the
    function: ``' '.join``

    Examples:

    >>> from itemloaders.processors import Join
    >>> proc = Join()
    >>> proc(['one', 'two', 'three'])
    'one two three'
    >>> proc = Join('<br>')
    >>> proc(['one', 'two', 'three'])
    'one<br>two<br>three'
    """

    def __init__(self, separator: str = " "):
        self.separator = separator

    def __call__(self, values: Any) -> str:
        return self.separator.join(values)

.. _async-loaders:

Asynchronous Item Loaders
=========================

.. currentmodule:: itemloaders

Some fields cannot be processed synchronously. For example, normalizing a
product category may require an asynchronous lookup in a dictionary service.
If you used a regular :class:`ItemLoader` for that, the processor would
return a coroutine object, and that coroutine object — not its result — would
be stored as the collected value.

:class:`AsyncItemLoader` is a drop-in asynchronous variant of
:class:`ItemLoader` that solves this: every method that runs processors is a
coroutine, and any processor — input, output, or ad-hoc — may be a regular
function, an ``async`` function, or a callable returning an awaitable. The
three kinds can be freely mixed; awaitable results are awaited in processor
declaration order and input value order, so coroutine objects never end up
stored as collected values.

.. note::

    :class:`AsyncItemLoader` never starts an event loop itself. Its methods
    are coroutines that you must ``await`` within your own asynchronous
    application (e.g. a Scrapy spider using ``async def parse``).

Example::

    import asyncio

    from itemloaders import AsyncItemLoader
    from itemloaders.processors import AsyncMapCompose, TakeFirst


    async def lookup_category(name):
        ...  # e.g. an asynchronous dictionary lookup
        return name.strip().title()


    class ProductLoader(AsyncItemLoader):
        name_in = AsyncMapCompose(lookup_category)
        name_out = TakeFirst()


    async def main():
        loader = ProductLoader()
        await loader.add_value("name", "  color tv ")
        return await loader.load_item()


    item = asyncio.run(main())
    # {'name': 'Color Tv'}

The asynchronous counterparts of the selector methods
(:meth:`~AsyncItemLoader.add_xpath`,
:meth:`~AsyncItemLoader.replace_xpath`, :meth:`~AsyncItemLoader.get_xpath`,
:meth:`~AsyncItemLoader.add_css`, :meth:`~AsyncItemLoader.replace_css`,
:meth:`~AsyncItemLoader.get_css`, :meth:`~AsyncItemLoader.add_jmes`,
:meth:`~AsyncItemLoader.replace_jmes` and :meth:`~AsyncItemLoader.get_jmes`)
are coroutines as well, as are :meth:`~AsyncItemLoader.get_value`,
:meth:`~AsyncItemLoader.add_value`, :meth:`~AsyncItemLoader.replace_value`,
:meth:`~AsyncItemLoader.get_output_value` and
:meth:`~AsyncItemLoader.load_item`.

Everything else works exactly as with :class:`ItemLoader`:

-   Nested loaders created with :meth:`~ItemLoader.nested_xpath` and
    :meth:`~ItemLoader.nested_css` share the collected values with their
    parent and inherit (and may override) its :ref:`Loader context
    <loaders-context>`.

-   Field metadata and class-level processors keep their usual precedence.

-   Errors are raised as :exc:`ValueError` chaining the original exception as
    their ``__cause__``, indicating the field and the processing stage
    (input, output or ad-hoc processor).

Asynchronous composition processors
-----------------------------------

:mod:`itemloaders.processors` provides asynchronous variants of the
composition processors, for use with :class:`AsyncItemLoader`:

-   :class:`~itemloaders.processors.AsyncCompose` works like
    :class:`~itemloaders.processors.Compose`, including the ``stop_on_none``
    short-circuit, but awaits functions that return awaitables.

-   :class:`~itemloaders.processors.AsyncMapCompose` works like
    :class:`~itemloaders.processors.MapCompose`, including ``None`` filtering
    and list flattening, but awaits functions that return awaitables.

Both accept any mix of regular functions, ``async`` functions and callables
returning awaitables::

    from itemloaders.processors import AsyncMapCompose

    async def to_upper(value):
        return value.upper()

    def to_awaitable(value):
        return some_async_client.normalize(value)  # returns an awaitable

    name_in = AsyncMapCompose(str.strip, to_upper, to_awaitable)

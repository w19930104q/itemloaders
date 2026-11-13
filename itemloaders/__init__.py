"""
Item Loader

See documentation in :ref:`declaring-loaders`.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from itemadapter import ItemAdapter
from parsel import Selector  # noqa: TC002  # for sphinx
from parsel.utils import extract_regex, flatten

from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
    from re import Pattern
    from typing import ClassVar


class _StatsCollector(Protocol):
    def inc_value(self, key: str, count: int = 1) -> None: ...


def unbound_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Allow to use single-argument functions as input or output processors
    (no need to define an unused first 'self' argument)
    """
    with suppress(AttributeError):
        if "." not in method.__qualname__:
            return method.__func__  # type: ignore[attr-defined, no-any-return]
    return method


@dataclass(frozen=True)
class DerivedField:
    """Declaration of a field whose value is computed from other fields.

    :param callback: Callable that receives the final output values of the
        fields listed in ``depends_on``, as positional arguments and in that
        order, and returns the raw value of the derived field. The returned
        value is then passed through the input and output processors declared
        for the derived field, exactly like a value collected with
        :meth:`ItemLoader.add_value`. The callback may additionally request the
        active :ref:`loader context <loaders-context>` through a
        ``loader_context`` keyword argument.
    :param depends_on: Names of the fields the callback depends on, in the
        order in which their final output values are passed to *callback*.
        A single field name may be given instead of a sequence.

    Derived fields are declared through the ``derived_fields`` class attribute
    of :class:`ItemLoader`. See :ref:`derived-fields`.
    """

    callback: Callable[..., Any]
    depends_on: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        # Normalize a single field name to a one-element tuple so the
        # declaration always iterates field names in dependency order.
        deps = self.depends_on
        object.__setattr__(
            self, "depends_on", (deps,) if isinstance(deps, str) else tuple(deps)
        )

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Field names passed to the callback, in order."""
        return cast("tuple[str, ...]", self.depends_on)


def _merge_derived_fields(
    loader_cls: type[ItemLoader],
) -> dict[str, DerivedField]:
    """Merge the :attr:`ItemLoader.derived_fields` mappings of the whole class
    hierarchy, so subclasses can add declarations and override inherited ones.
    """
    declarations: dict[str, DerivedField] = {}
    # Base classes first, each subclass overriding declarations of the same name.
    for base in reversed(loader_cls.__mro__):
        declarations.update(getattr(base, "derived_fields", None) or {})
    return declarations


class ItemLoader:
    """
    Return a new Item Loader for populating the given item. If no item is
    given, one is instantiated automatically using the class in
    :attr:`default_item_class`.

    When instantiated with a ``selector`` parameter the :class:`ItemLoader` class
    provides convenient mechanisms for extracting data from web pages
    using parsel_ selectors.

    :param item: The item instance to populate using subsequent calls to
        :meth:`~ItemLoader.add_xpath`, :meth:`~ItemLoader.add_css`,
        :meth:`~ItemLoader.add_jmes` or :meth:`~ItemLoader.add_value`.
    :type item: :class:`dict` object

    :param selector: The selector to extract data from, when using the
        :meth:`add_xpath` (resp. :meth:`add_css`, :meth:`add_jmes`) or :meth:`replace_xpath`
        (resp. :meth:`replace_css`, :meth:`replace_jmes`) method.
    :type selector: :class:`~parsel.selector.Selector` object

    The item, selector and the remaining keyword arguments are
    assigned to the Loader context (accessible through the :attr:`context` attribute).

    Pass *stats* to keep track of which parsing rules are being used, as
    described in :ref:`rule-usage`.

    .. attribute:: item

        The item object being parsed by this Item Loader.
        This is mostly used as a property so when attempting to override this
        value, you may want to check out :attr:`default_item_class` first.

    .. attribute:: context

        The currently active :ref:`Context <loaders-context>` of this Item Loader.
        Refer to :ref:`loaders-context` for more information about the Loader Context.

    .. attribute:: default_item_class

        An Item class (or factory), used to instantiate items when not given in
        the ``__init__`` method.

        .. versionchanged:: VERSION
            The field defaults of an item instantiated this way are no longer
            treated as loaded values.

        .. warning:: Currently, this factory/class needs to be
            callable/instantiated without any arguments.
            If you are using ``dataclasses``, please consider the following
            alternative::

                from dataclasses import dataclass, field
                from typing import Optional

                @dataclass
                class Product:
                    name: Optional[str] = field(default=None)
                    price: Optional[float] = field(default=None)

    .. attribute:: default_input_processor

        The default input processor to use for those fields which don't specify
        one.

    .. attribute:: default_output_processor

        The default output processor to use for those fields which don't specify
        one.

    .. attribute:: selector

        The :class:`~parsel.selector.Selector` object to extract data from.
        It's the selector given in the ``__init__`` method.
        This attribute is meant to be read-only.

    .. _parsel: https://parsel.readthedocs.io/en/latest/
    """

    default_item_class: type = dict
    default_input_processor: Callable[..., Any] = Identity()
    default_output_processor: Callable[..., Any] = Identity()

    #: Class-level mapping of derived field names to their
    #: :class:`DerivedField` declarations. Subclasses inherit the declarations
    #: of their parents and may add new ones or override inherited ones.
    derived_fields: ClassVar[Mapping[str, DerivedField]] = {}

    def __init__(
        self,
        item: Any = None,
        selector: Selector | None = None,
        parent: ItemLoader | None = None,
        stats: _StatsCollector | None = None,
        **context: Any,
    ):
        self.selector: Selector | None = selector
        self.stats: _StatsCollector | None = (
            stats if stats is not None or parent is None else parent.stats
        )
        context.update(selector=selector)
        # An item built here has no data, only the field defaults of its class,
        # which must not be mistaken for loaded values.
        has_initial_values = item is not None
        if item is None:
            item = self.default_item_class()
        self._local_item = item
        context["item"] = item
        self.context: MutableMapping[str, Any] = context
        self.parent: ItemLoader | None = parent
        self._local_values: dict[str, list[Any]] = {}
        # Merge the class hierarchy once per loader instance. Nested loaders
        # are created from the same class, so each one caches the same mapping.
        self._derived_fields_map: Mapping[str, DerivedField] = _merge_derived_fields(
            self.__class__
        )
        if has_initial_values:
            for field_name, value in ItemAdapter(item).items():
                if field_name in self._derived_fields_map:
                    raise self._derived_collection_error(field_name)
                self._values.setdefault(field_name, [])
                self._values[field_name] += arg_to_iter(value)

    @property
    def _values(self) -> dict[str, list[Any]]:
        if self.parent is not None:
            return self.parent._values
        return self._local_values

    @property
    def item(self) -> Any:
        if self.parent is not None:
            return self.parent.item
        return self._local_item

    def nested_xpath(self, xpath: str, **context: Any) -> Self:
        """
        Create a nested loader with an xpath selector.
        The supplied selector is applied relative to selector associated
        with this :class:`ItemLoader`. The nested loader shares the item
        with the parent :class:`ItemLoader` so calls to :meth:`add_xpath`,
        :meth:`add_value`, :meth:`replace_value`, etc. will behave as expected.
        """
        self._check_selector_method()
        assert self.selector is not None
        selector = self.selector.xpath(xpath)
        context = self._get_nested_context(context)
        context.update(selector=selector)
        return self.__class__(item=self.item, parent=self, **context)

    def nested_css(self, css: str, **context: Any) -> Self:
        """
        Create a nested loader with a css selector.
        The supplied selector is applied relative to selector associated
        with this :class:`ItemLoader`. The nested loader shares the item
        with the parent :class:`ItemLoader` so calls to :meth:`add_xpath`,
        :meth:`add_value`, :meth:`replace_value`, etc. will behave as expected.
        """
        self._check_selector_method()
        assert self.selector is not None
        selector = self.selector.css(css)
        context = self._get_nested_context(context)
        context.update(selector=selector)
        return self.__class__(item=self.item, parent=self, **context)

    def _get_nested_context(self, context: dict[str, Any]) -> dict[str, Any]:
        inherited = {key: value for key, value in self.context.items() if key != "item"}
        inherited.update(context)
        return inherited

    def add_value(
        self,
        field_name: str | None,
        value: Any,
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Process and then add the given ``value`` for the given field.

        The value is first passed through :meth:`get_value` by giving the
        ``processors`` and ``kwargs``, and then passed through the
        :ref:`field input processor <processors>` and its result
        appended to the data collected for that field. If the field already
        contains collected data, the new data is added.

        The given ``field_name`` can be ``None``, in which case values for
        multiple fields may be added. And the processed value should be a dict
        with field_name mapped to values.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        Examples::

            loader.add_value('name', 'Color TV')
            loader.add_value('colours', ['white', 'blue'])
            loader.add_value('length', '100')
            loader.add_value('name', 'name: foo', TakeFirst(), re='name: (.+)')
            loader.add_value(None, {'name': 'foo', 'sex': 'male'})

        """
        value = self.get_value(value, *processors, re=re, **kw)
        if value is None:
            return self
        if not field_name:
            for k, v in value.items():
                self._add_value(k, v)
        else:
            self._add_value(field_name, value)
        return self

    def replace_value(
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

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader
        """
        value = self.get_value(value, *processors, re=re, **kw)
        if value is None:
            return self
        if not field_name:
            for k, v in value.items():
                self._replace_value(k, v)
        else:
            self._replace_value(field_name, value)
        return self

    def _add_value(self, field_name: str, value: Any) -> None:
        self._check_not_derived(field_name)
        self._add_value_checked(field_name, value)

    def _add_value_checked(self, field_name: str, value: Any) -> None:
        value = arg_to_iter(value)
        processed_value = self._process_input_value(field_name, value)
        if processed_value:
            self._values.setdefault(field_name, [])
            self._values[field_name] += arg_to_iter(processed_value)

    def _replace_value(self, field_name: str, value: Any) -> None:
        self._check_not_derived(field_name)
        self._values.pop(field_name, None)
        self._add_value_checked(field_name, value)

    def _check_not_derived(self, field_name: str) -> None:
        if field_name in self._derived_fields():
            raise self._derived_collection_error(field_name)

    def _derived_collection_error(self, field_name: str) -> ValueError:
        return ValueError(
            f"Field {field_name!r} is a derived field declared in "
            f"derived_fields, its value is computed and must not be collected "
            f"explicitly. Remove the add_*/replace_* call or initial item "
            f"value targeting it, or rename the derived field declaration."
        )

    def _derived_fields(self) -> Mapping[str, DerivedField]:
        return self._derived_fields_map

    def get_value(
        self,
        value: Any,
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Process the given ``value`` by the given ``processors`` and keyword
        arguments.

        Available keyword arguments:

        :param re: a regular expression to use for extracting data from the
            given value using :func:`~parsel.utils.extract_regex` method,
            applied before processors
        :type re: str or typing.Pattern[str]

        Examples:

        >>> from itemloaders import ItemLoader
        >>> from itemloaders.processors import TakeFirst
        >>> loader = ItemLoader()
        >>> loader.get_value('name: foo', TakeFirst(), str.upper, re='name: (.+)')
        'FOO'
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
                value = proc(value)
            except Exception as e:
                raise ValueError(
                    f"Error with processor {_proc.__class__.__name__} "
                    f"value={value!r} error='{type(e).__name__}: {e!s}'"
                ) from e
        return value

    def load_item(self) -> Any:
        """
        Populate the item with the data collected so far, and return it. The
        data collected is first passed through the :ref:`output processors
        <processors>` to get the final value to assign to each item field.

        Fields declared in :attr:`derived_fields` are computed first, each at
        most once, from the final output values of their dependencies. Every
        field output is computed before the item is written to, so a failing
        processor or derived field callback never leaves a partially
        populated item.
        """
        adapter = ItemAdapter(self.item)
        memo: dict[str, Any] = {}
        # Validate the whole derived field dependency graph before running any
        # processor, so cycles and unknown dependencies fail with their path.
        # Declaration order keeps the reported dependency path deterministic.
        self._validate_derived_graph(tuple(self._derived_fields()))
        # Collected fields first, in their insertion order, so the write order
        # of non-derived fields is unchanged. Resolving them also primes the
        # memo for any derived field that depends on collected values. A
        # derived field name can never reach ``_values``: collecting one is
        # rejected up front by ``_check_not_derived``.
        computed: dict[str, Any] = {}
        for field_name in tuple(self._values):
            computed[field_name] = self._resolve_output_value(field_name, memo)
        # Derived fields are resolved in declaration order; dependencies are
        # pulled from the memo when a shared dependency (a diamond) was already
        # evaluated, so every field output is computed at most once.
        for field_name in self._derived_fields():
            computed[field_name] = self._resolve_output_value(field_name, memo)
        for field_name, value in computed.items():
            if value is not None:
                adapter[field_name] = value

        return adapter.item

    def get_output_value(self, field_name: str) -> Any:
        """
        Return the collected values parsed using the output processor, for the
        given field. This method doesn't populate or modify the item at all.

        For a :attr:`derived field <derived_fields>`, only the fields it
        depends on are computed, and the computation is not stored on the
        item.
        """
        memo: dict[str, Any] = {}
        self._validate_derived_graph((field_name,))
        return self._resolve_output_value(field_name, memo)

    def _resolve_output_value(self, field_name: str, memo: dict[str, Any]) -> Any:
        """Return the final output value of *field_name*, evaluating derived
        fields recursively and memoizing every result within *memo*.
        """
        if field_name in memo:
            return memo[field_name]
        declaration = self._derived_fields().get(field_name)
        if declaration is None:
            value = self._run_output_processor(field_name)
            memo[field_name] = value
            return value
        dependency_values = [
            self._resolve_output_value(dependency, memo)
            for dependency in declaration.dependencies
        ]
        callback = wrap_loader_context(declaration.callback, self.context)
        try:
            derived_value = callback(*dependency_values)
        except Exception as e:
            raise ValueError(
                f"Error with derived field callback: field={field_name!r} "
                f"dependencies={list(declaration.dependencies)!r} "
                f"error='{type(e).__name__}: {e!s}'"
            ) from e
        value = self._process_derived_value(field_name, derived_value)
        memo[field_name] = value
        return value

    def _process_derived_value(self, field_name: str, derived_value: Any) -> Any:
        """Run the derived field's input processor, then its output processor,
        mirroring what a collected value goes through.
        """
        collected = self._process_input_value(
            field_name, list(arg_to_iter(derived_value))
        )
        return self._run_output_processor(field_name, list(arg_to_iter(collected)))

    def _run_output_processor(self, field_name: str, value: Any | None = None) -> Any:
        proc = self.get_output_processor(field_name)
        proc = wrap_loader_context(proc, self.context)
        if value is None:
            value = self._values.get(field_name, [])
        try:
            return proc(value)
        except Exception as e:
            raise ValueError(
                f"Error with output processor: field={field_name!r} "
                f"value={value!r} error='{type(e).__name__}: {e!s}'"
            ) from e

    def _validate_derived_graph(self, roots: Sequence[str]) -> None:
        """Walk the dependency graph reachable from *roots* and raise a
        :class:`ValueError` describing the full dependency path on the first
        cycle or unknown dependency, before any processor runs.

        *roots* order is preserved so the reported path is deterministic.
        """
        declarations = self._derived_fields()
        if not declarations:
            # No derived fields: nothing can be cyclic or unknown through a
            # dependency, and the hot path must stay allocation-free.
            return
        adapter = ItemAdapter(self.item)
        # Schema-less items (e.g. plain dicts) accept any field name, so an
        # uncollected dependency is treated as an empty one rather than an
        # unknown one. Schema-bound items (dataclasses, Scrapy items, ...)
        # only allow names declared by their fields.
        known_item_fields = set(adapter.field_names())
        open_item = not known_item_fields

        def is_known(field_name: str) -> bool:
            return (
                open_item
                or field_name in declarations
                or field_name in known_item_fields
            )

        def visit(field_name: str, path: tuple[str, ...]) -> None:
            declaration = declarations.get(field_name)
            if declaration is None:
                # Only names reached through a dependency can be unknown: a
                # plain field requested directly is always a valid request,
                # even when the loader declares no derived field at all.
                if path and not is_known(field_name):
                    raise ValueError(
                        "Unknown derived field dependency: "
                        + " -> ".join([*path, field_name])
                    )
                return
            if field_name in path:
                cycle_start = path.index(field_name)
                cycle = [*path[cycle_start:], field_name]
                raise ValueError(
                    "Cyclic derived field dependency detected: " + " -> ".join(cycle)
                )
            current_path = (*path, field_name)
            for dependency in declaration.dependencies:
                visit(dependency, current_path)

        for root in roots:
            visit(root, ())

    def get_collected_values(self, field_name: str) -> list[Any]:
        """Return the collected values for the given field."""
        return self._values.get(field_name, [])

    def get_input_processor(self, field_name: str) -> Callable[..., Any]:
        proc = getattr(self, f"{field_name}_in", None)
        if not proc:
            proc = self._get_item_field_attr(
                field_name, "input_processor", self.default_input_processor
            )
        return unbound_method(proc)

    def get_output_processor(self, field_name: str) -> Callable[..., Any]:
        proc = getattr(self, f"{field_name}_out", None)
        if not proc:
            proc = self._get_item_field_attr(
                field_name, "output_processor", self.default_output_processor
            )
        return unbound_method(proc)

    def _get_item_field_attr(
        self, field_name: str, key: Any, default: Any = None
    ) -> Any:
        field_meta = ItemAdapter(self.item).get_field_meta(field_name)
        return field_meta.get(key, default)

    def _process_input_value(self, field_name: str, value: Any) -> Any:
        proc = self.get_input_processor(field_name)
        _proc = proc
        proc = wrap_loader_context(proc, self.context)
        try:
            return proc(value)
        except Exception as e:
            raise ValueError(
                f"Error with input processor {_proc.__class__.__name__}:"
                f" field={field_name!r} value={value!r} error='{type(e).__name__}: {e!s}'"
            ) from e

    def _track_rule(
        self, field_name: str | None, rule_type: str, rule: str, values: list[Any]
    ) -> list[Any]:
        if self.stats is not None:
            key = f"parser/{field_name or '<none>'}/{rule_type}/{rule}"
            self.stats.inc_value(key, 1 if values else 0)
        return values

    def _check_selector_method(self) -> None:
        if self.selector is None:
            raise RuntimeError(
                f"To use XPath or CSS selectors, {self.__class__.__name__} "
                f"must be instantiated with a selector"
            )

    def add_xpath(
        self,
        field_name: str | None,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`ItemLoader.add_value` but receives an XPath instead of a
        value, which is used to extract a list of strings from the
        selector associated with this :class:`ItemLoader`.

        *xpath* may also be an iterable of XPath expressions, in which case the
        strings extracted by all of them are collected.

        See :meth:`get_xpath` for ``kwargs``.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        Examples::

            # HTML snippet: <p class="product-name">Color TV</p>
            loader.add_xpath('name', '//p[@class="product-name"]')
            # HTML snippet: <p id="price">the price is $1200</p>
            loader.add_xpath('price', '//p[@id="price"]', re='the price is (.*)')

        """
        values = self._get_xpathvalues(xpath, field_name, **kw)
        return self.add_value(field_name, values, *processors, re=re, **kw)

    def replace_xpath(
        self,
        field_name: str | None,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_xpath` but replaces collected data instead of adding it.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        """
        values = self._get_xpathvalues(xpath, field_name, **kw)
        return self.replace_value(field_name, values, *processors, re=re, **kw)

    def get_xpath(
        self,
        xpath: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Similar to :meth:`ItemLoader.get_value` but receives an XPath instead of a
        value, which is used to extract a list of strings from the
        selector associated with this :class:`ItemLoader`.

        *xpath* may also be an iterable of XPath expressions, in which case the
        strings extracted by all of them are collected.

        *re* is a regular expression, as a string or a compiled pattern, used to
        further extract data from the selected XPath region.

        Examples::

            # HTML snippet: <p class="product-name">Color TV</p>
            loader.get_xpath('//p[@class="product-name"]')
            # HTML snippet: <p id="price">the price is $1200</p>
            loader.get_xpath('//p[@id="price"]', TakeFirst(), re='the price is (.*)')

        """
        values = self._get_xpathvalues(xpath, **kw)
        return self.get_value(values, *processors, re=re, **kw)

    def _get_xpathvalues(
        self,
        xpaths: str | Iterable[str],
        field_name: str | None = None,
        **kw: Any,
    ) -> list[Any]:
        self._check_selector_method()
        assert self.selector is not None
        xpaths = arg_to_iter(xpaths)
        return flatten(
            self._track_rule(
                field_name, "xpath", xpath, self.selector.xpath(xpath, **kw).getall()
            )
            for xpath in xpaths
        )

    def add_css(
        self,
        field_name: str | None,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`ItemLoader.add_value` but receives a CSS selector
        instead of a value, which is used to extract a list of strings
        from the selector associated with this :class:`ItemLoader`.

        *css* may also be an iterable of CSS selectors, in which case the
        strings extracted by all of them are collected.

        See :meth:`get_css` for ``kwargs``.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        Examples::

            # HTML snippet: <p class="product-name">Color TV</p>
            loader.add_css('name', 'p.product-name')
            # HTML snippet: <p id="price">the price is $1200</p>
            loader.add_css('price', 'p#price', re='the price is (.*)')

        """
        values = self._get_cssvalues(css, field_name)
        return self.add_value(field_name, values, *processors, re=re, **kw)

    def replace_css(
        self,
        field_name: str | None,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_css` but replaces collected data instead of adding it.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        """
        values = self._get_cssvalues(css, field_name)
        return self.replace_value(field_name, values, *processors, re=re, **kw)

    def get_css(
        self,
        css: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Similar to :meth:`ItemLoader.get_value` but receives a CSS selector
        instead of a value, which is used to extract a list of strings
        from the selector associated with this :class:`ItemLoader`.

        *css* may also be an iterable of CSS selectors, in which case the
        strings extracted by all of them are collected.

        *re* is a regular expression, as a string or a compiled pattern, used to
        further extract data from the selected CSS region.

        Examples::

            # HTML snippet: <p class="product-name">Color TV</p>
            loader.get_css('p.product-name')
            # HTML snippet: <p id="price">the price is $1200</p>
            loader.get_css('p#price', TakeFirst(), re='the price is (.*)')
        """
        values = self._get_cssvalues(css)
        return self.get_value(values, *processors, re=re, **kw)

    def _get_cssvalues(
        self, csss: str | Iterable[str], field_name: str | None = None
    ) -> list[Any]:
        self._check_selector_method()
        assert self.selector is not None
        csss = arg_to_iter(csss)
        return flatten(
            self._track_rule(field_name, "css", css, self.selector.css(css).getall())
            for css in csss
        )

    def add_jmes(
        self,
        field_name: str | None,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`ItemLoader.add_value` but receives a JMESPath selector
        instead of a value, which is used to extract a list of strings
        from the selector associated with this :class:`ItemLoader`.

        *jmes* may also be an iterable of JMESPath selectors, in which case the
        strings extracted by all of them are collected.

        See :meth:`get_jmes` for ``kwargs``.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader

        Examples::

            # JSON snippet: {"name": "Color TV"}
            loader.add_jmes('name', 'name')
            # JSON snippet: {"price": "the price is $1200"}
            loader.add_jmes('price', 'price', TakeFirst(), re='the price is (.*)')
        """
        values = self._get_jmesvalues(jmes, field_name)
        return self.add_value(field_name, values, *processors, re=re, **kw)

    def replace_jmes(
        self,
        field_name: str | None,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Self:
        """
        Similar to :meth:`add_jmes` but replaces collected data instead of adding it.

        :returns: The current ItemLoader instance for method chaining.
        :rtype: ItemLoader
        """
        values = self._get_jmesvalues(jmes, field_name)
        return self.replace_value(field_name, values, *processors, re=re, **kw)

    def get_jmes(
        self,
        jmes: str | Iterable[str],
        *processors: Callable[..., Any],
        re: str | Pattern[str] | None = None,
        **kw: Any,
    ) -> Any:
        """
        Similar to :meth:`ItemLoader.get_value` but receives a JMESPath selector
        instead of a value, which is used to extract a list of strings
        from the selector associated with this :class:`ItemLoader`.

        *jmes* may also be an iterable of JMESPath selectors, in which case the
        strings extracted by all of them are collected.

        *re* is a regular expression, as a string or a compiled pattern, used to
        further extract data from the selected JMESPath.

        Examples::

            # JSON snippet: {"name": "Color TV"}
            loader.get_jmes('name')
            # JSON snippet: {"price": "the price is $1200"}
            loader.get_jmes('price', TakeFirst(), re='the price is (.*)')
        """
        values = self._get_jmesvalues(jmes)
        return self.get_value(values, *processors, re=re, **kw)

    def _get_jmesvalues(
        self, jmess: str | Iterable[str], field_name: str | None = None
    ) -> list[Any]:
        self._check_selector_method()
        assert self.selector is not None
        jmess = arg_to_iter(jmess)
        if not hasattr(self.selector, "jmespath"):
            raise AttributeError(
                "Please install parsel >= 1.8.1 to get JMESPath support"
            )
        return flatten(
            self._track_rule(
                field_name, "jmes", jmes, self.selector.jmespath(jmes).getall()
            )
            for jmes in jmess
        )

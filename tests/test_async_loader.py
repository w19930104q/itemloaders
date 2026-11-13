import asyncio
import gc
import inspect
import json
from collections.abc import Coroutine, Generator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pytest
from parsel import Selector

from itemloaders import AsyncItemLoader, ItemLoader
from itemloaders.processors import AsyncCompose, AsyncMapCompose, Identity, TakeFirst

T = TypeVar("T")

needs_parsel_jmespath = pytest.mark.skipif(
    not hasattr(Selector, "jmespath"), reason="parsel < 1.8.1"
)

# Fail the test if a coroutine is left un-awaited (RuntimeWarning from the GC)
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

HTML = """
<html>
<body>
<div id="id">marta</div>
<p class="price">the price is $1200</p>
<a href="http://www.scrapy.org">homepage</a>
</body>
</html>
"""

JSON_DATA = json.dumps(
    {
        "name": "marta",
        "price": "the price is $1200",
        "website": {"url": "http://www.scrapy.org"},
    }
)


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine in a fresh event loop owned by the test."""
    try:
        return asyncio.run(coro)
    finally:
        # Trigger the GC so that any coroutine left un-awaited raises a
        # RuntimeWarning, turned into an error by the module-level marker.
        gc.collect()


class ControlledAwaitable:
    """A minimal awaitable that is not a coroutine."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.was_awaited = False

    def __await__(self) -> Generator[Any, None, Any]:
        self.was_awaited = True
        yield from asyncio.sleep(0).__await__()
        return self.value


def sync_processor(value: list[str]) -> list[str]:
    return [v.strip() for v in value]


async def async_processor(value: list[str]) -> list[str]:
    await asyncio.sleep(0)
    return [v.upper() for v in value]


def awaitable_returning_processor(value: list[str]) -> ControlledAwaitable:
    return ControlledAwaitable([v[::-1] for v in value])


class TestMixedProcessors:
    def test_get_value_mixes_sync_async_and_awaitable_processors(self):
        loader = AsyncItemLoader()
        result = run(
            loader.get_value(
                [" a ", "b "],
                sync_processor,
                async_processor,
                awaitable_returning_processor,
            )
        )
        assert result == ["A", "B"]

    def test_processors_awaited_in_declaration_and_input_order(self):
        calls = []

        async def first(v):
            await asyncio.sleep(0)
            calls.append(("first", v))
            return v

        def second(v):
            calls.append(("second", v))
            return v

        loader = AsyncItemLoader()
        run(loader.get_value(["x", "y"], AsyncMapCompose(first, second)))
        assert calls == [
            ("first", "x"),
            ("first", "y"),
            ("second", "x"),
            ("second", "y"),
        ]

    def test_awaitable_is_awaited_not_stored(self):
        loader = AsyncItemLoader()
        awaitable = ControlledAwaitable("value")

        def processor(value):
            return awaitable

        result = run(loader.get_value("ignored", processor))
        assert result == "value"
        assert awaitable.was_awaited

    def test_add_value_with_async_adhoc_processors(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", [" marta "], AsyncMapCompose(str.strip)))
        assert loader.get_collected_values("name") == ["marta"]

    def test_add_value_none_field_with_dict(self):
        loader = AsyncItemLoader()
        run(loader.add_value(None, {"name": "foo", "sex": "male"}))
        assert loader.get_collected_values("name") == ["foo"]
        assert loader.get_collected_values("sex") == ["male"]

    def test_add_value_none_result_is_skipped(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", "x", lambda v: None))
        assert loader.get_collected_values("name") == []

    def test_replace_value(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", "marta"))
        run(loader.replace_value("name", "flash", AsyncMapCompose(str.upper)))
        assert loader.get_collected_values("name") == ["FLASH"]

    def test_replace_value_none_field_with_dict(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", "marta"))
        run(loader.add_value("sex", "unknown"))
        run(loader.replace_value(None, {"name": "flash"}))
        assert loader.get_collected_values("name") == ["flash"]
        assert loader.get_collected_values("sex") == ["unknown"]

    def test_duplicate_values_are_collected(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", "marta"))
        run(loader.add_value("name", "marta"))
        assert loader.get_collected_values("name") == ["marta", "marta"]

    def test_methods_return_awaitables(self):
        loader = AsyncItemLoader()
        coro = loader.add_value("name", "marta")
        assert inspect.iscoroutine(coro)
        run(coro)
        assert loader.get_collected_values("name") == ["marta"]


async def async_upper(v: str) -> str:
    await asyncio.sleep(0)
    return v.upper()


class AsyncInputLoader(AsyncItemLoader):
    name_in = AsyncMapCompose(str.strip, async_upper)


class AsyncOutputLoader(AsyncItemLoader):
    name_out = AsyncCompose(async_processor, TakeFirst())


class TestInputOutputProcessors:
    def test_async_input_processor(self):
        loader = AsyncInputLoader()
        run(loader.add_value("name", " marta"))
        assert loader.get_collected_values("name") == ["MARTA"]

    def test_async_output_processor(self):
        loader = AsyncOutputLoader()
        run(loader.add_value("name", "marta"))
        assert run(loader.get_output_value("name")) == "MARTA"

    def test_load_item_with_async_processors(self):
        loader = AsyncInputLoader()
        run(loader.add_value("name", " marta "))
        assert run(loader.load_item()) == {"name": ["MARTA"]}

    def test_load_item_populates_the_same_dict_item(self):
        item = {"summary": "lala"}
        loader = AsyncItemLoader(item=item)
        run(loader.add_value("name", "marta"))
        loaded = run(loader.load_item())
        assert loaded is item
        assert item == {"summary": ["lala"], "name": ["marta"]}

    def test_load_item_with_dataclass_item(self):
        @dataclass
        class Product:
            name: Any = ""
            price: Any = ""

        loader = AsyncInputLoader(item=Product(name="ignored", price="1"))
        run(loader.add_value("name", " marta "))
        run(loader.add_value("price", "100"))
        loaded = run(loader.load_item())
        assert loaded == Product(name=["ignored", "MARTA"], price=["1", "100"])

    def test_default_processors_still_work(self):
        loader = AsyncItemLoader()
        assert isinstance(loader.get_input_processor("name"), Identity)
        run(loader.add_value("name", "marta"))
        assert run(loader.load_item()) == {"name": ["marta"]}

    def test_field_metadata_and_class_processor_priority(self):
        meta_in = AsyncMapCompose(lambda v: v + "-meta")
        class_in = AsyncMapCompose(lambda v: v + "-class")

        @dataclass
        class Product:
            name: str = field(default="", metadata={"input_processor": meta_in})
            price: str = field(default="", metadata={"input_processor": meta_in})

        class ProductLoader(AsyncItemLoader):
            default_item_class = Product
            name_in = class_in

        loader = ProductLoader()
        run(loader.add_value("name", "x"))
        run(loader.add_value("price", "y"))
        # The class-level processor takes precedence over field metadata,
        # while fields without a class-level processor use the metadata one.
        assert loader.get_collected_values("name") == ["x-class"]
        assert loader.get_collected_values("price") == ["y-meta"]


class TestSelectors:
    selector = Selector(text=HTML)
    jmes_selector = Selector(text=JSON_DATA)

    def test_add_xpath(self):
        loader = AsyncItemLoader(selector=self.selector)
        run(loader.add_xpath("name", "//div[@id='id']/text()"))
        assert loader.get_collected_values("name") == ["marta"]

    def test_add_xpath_with_async_processor_and_re(self):
        loader = AsyncItemLoader(selector=self.selector)
        run(
            loader.add_xpath(
                "price",
                "//p[@class='price']/text()",
                AsyncMapCompose(str.upper),
                re="the price is (.*)",
            )
        )
        assert loader.get_collected_values("price") == ["$1200"]

    def test_get_xpath(self):
        loader = AsyncItemLoader(selector=self.selector)
        result = run(loader.get_xpath("//a/@href", TakeFirst()))
        assert result == "http://www.scrapy.org"

    def test_replace_xpath(self):
        loader = AsyncItemLoader(selector=self.selector)
        run(loader.add_value("name", "flash"))
        run(loader.replace_xpath("name", "//div[@id='id']/text()"))
        assert loader.get_collected_values("name") == ["marta"]

    def test_add_css(self):
        loader = AsyncItemLoader(selector=self.selector)
        run(loader.add_css("name", "div#id::text", AsyncMapCompose(str.title)))
        assert loader.get_collected_values("name") == ["Marta"]

    def test_get_css_with_re(self):
        loader = AsyncItemLoader(selector=self.selector)
        result = run(
            loader.get_css("p.price::text", TakeFirst(), re="the price is (.*)")
        )
        assert result == "$1200"

    def test_replace_css(self):
        loader = AsyncItemLoader(selector=self.selector)
        run(loader.add_value("name", "flash"))
        run(loader.replace_css("name", "div#id::text"))
        assert loader.get_collected_values("name") == ["marta"]

    @needs_parsel_jmespath
    def test_add_jmes(self):
        loader = AsyncItemLoader(selector=self.jmes_selector)
        run(loader.add_jmes("name", "name", AsyncMapCompose(str.upper)))
        assert loader.get_collected_values("name") == ["MARTA"]

    @needs_parsel_jmespath
    def test_get_jmes(self):
        loader = AsyncItemLoader(selector=self.jmes_selector)
        result = run(loader.get_jmes("website.url", TakeFirst()))
        assert result == "http://www.scrapy.org"

    @needs_parsel_jmespath
    def test_replace_jmes(self):
        loader = AsyncItemLoader(selector=self.jmes_selector)
        run(loader.add_value("name", "flash"))
        run(loader.replace_jmes("name", "name"))
        assert loader.get_collected_values("name") == ["marta"]

    def test_selector_methods_require_a_selector(self):
        loader = AsyncItemLoader()
        with pytest.raises(RuntimeError):
            run(loader.add_xpath("name", "//div"))
        with pytest.raises(RuntimeError):
            run(loader.add_css("name", "div"))
        with pytest.raises(RuntimeError):
            run(loader.get_xpath("//div"))


def context_processor(
    value: list[str], loader_context: dict[str, Any] | None = None
) -> list[str]:
    assert loader_context is not None
    return [f"{v}-{loader_context['key']}" for v in value]


class ContextLoader(AsyncItemLoader):
    name_in = AsyncMapCompose(context_processor)


class TestNestedLoaders:
    selector = Selector(text=HTML)

    def test_nested_loader_shares_collected_values(self):
        loader = AsyncItemLoader(selector=self.selector)
        nested = loader.nested_xpath("//body")
        run(nested.add_xpath("name", ".//div[@id='id']/text()"))
        run(loader.add_value("price", "100"))
        assert isinstance(nested, AsyncItemLoader)
        assert loader.get_collected_values("name") == ["marta"]
        assert run(loader.load_item()) == {"name": ["marta"], "price": ["100"]}

    def test_nested_css_loader(self):
        loader = AsyncItemLoader(selector=self.selector)
        nested = loader.nested_css("body")
        run(nested.add_css("name", "div#id::text"))
        assert loader.get_collected_values("name") == ["marta"]

    def test_nested_loader_inherits_and_overrides_context(self):
        loader = ContextLoader(selector=self.selector, key="parent")
        nested = loader.nested_xpath("//body")
        run(nested.add_value("name", "x"))
        overridden = loader.nested_xpath("//body", key="child")
        run(overridden.add_value("name", "y"))
        assert loader.get_collected_values("name") == ["x-parent", "y-child"]


class TestAsyncCompose:
    def test_mixes_sync_and_async_functions(self):
        async def to_upper(v):
            await asyncio.sleep(0)
            return v.upper()

        proc = AsyncCompose(lambda v: v[0], to_upper)
        assert run(proc(["hello", "world"])) == "HELLO"

    def test_stop_on_none_short_circuits(self):
        called = False

        async def should_not_run(v):
            nonlocal called
            called = True
            return v

        proc = AsyncCompose(lambda v: None, should_not_run)
        assert run(proc(["x"])) is None
        assert not called

    def test_stop_on_none_false_continues(self):
        async def replace_none(v):
            return "recovered" if v is None else v

        proc = AsyncCompose(lambda v: None, replace_none, stop_on_none=False)
        assert run(proc(["x"])) == "recovered"

    def test_awaitable_returning_function(self):
        proc = AsyncCompose(lambda v: ControlledAwaitable(v * 2))
        assert run(proc(21)) == 42

    def test_loader_context_is_passed(self):
        async def with_context(value, loader_context=None):
            return loader_context["key"]

        proc = AsyncCompose(with_context)
        assert run(proc("x", loader_context={"key": "value"})) == "value"

    def test_error_keeps_cause(self):
        async def broken(v):
            raise KeyError("boom")

        proc = AsyncCompose(broken)
        with pytest.raises(ValueError, match="Error in AsyncCompose") as exc_info:
            run(proc("x"))
        assert isinstance(exc_info.value.__cause__, KeyError)


class TestAsyncMapCompose:
    def test_filters_none_and_flattens_lists(self):
        async def filter_b(v):
            await asyncio.sleep(0)
            return None if v == "b" else [v, v.upper()]

        proc = AsyncMapCompose(filter_b)
        assert run(proc(["a", "b", "c"])) == ["a", "A", "c", "C"]

    def test_mixes_sync_async_and_awaitable_functions(self):
        proc = AsyncMapCompose(
            str.strip,
            lambda v: ControlledAwaitable(v.upper()),
            lambda v: v[::-1],
        )
        assert run(proc([" ab ", "cd "])) == ["BA", "DC"]

    def test_loader_context_is_passed(self):
        proc = AsyncMapCompose(context_processor)
        assert run(proc(["x"], loader_context={"key": "k"})) == ["x-k"]

    def test_error_keeps_cause(self):
        async def broken(v):
            raise KeyError("boom")

        proc = AsyncMapCompose(broken)
        with pytest.raises(ValueError, match="Error in AsyncMapCompose") as exc_info:
            run(proc(["x"]))
        assert isinstance(exc_info.value.__cause__, KeyError)


class TestExceptions:
    def test_input_processor_error_reports_field_and_stage(self):
        async def broken(values):
            raise KeyError("boom")

        class BrokenLoader(AsyncItemLoader):
            name_in = AsyncMapCompose(broken)

        loader = BrokenLoader()
        with pytest.raises(ValueError, match="input processor") as exc_info:
            run(loader.add_value("name", "marta"))
        message = str(exc_info.value)
        assert "input processor" in message
        assert "field='name'" in message
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert isinstance(exc_info.value.__cause__.__cause__, KeyError)

    def test_output_processor_error_reports_field_and_stage(self):
        async def broken(values):
            raise KeyError("boom")

        class BrokenLoader(AsyncItemLoader):
            name_out = AsyncCompose(broken)

        loader = BrokenLoader()
        run(loader.add_value("name", "marta"))
        with pytest.raises(ValueError, match="output processor") as exc_info:
            run(loader.load_item())
        message = str(exc_info.value)
        assert "output processor" in message
        assert "field='name'" in message
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert isinstance(exc_info.value.__cause__.__cause__, KeyError)

    def test_adhoc_processor_error_keeps_cause(self):
        async def broken(value):
            raise KeyError("boom")

        loader = AsyncItemLoader()
        with pytest.raises(ValueError, match="Error with processor") as exc_info:
            run(loader.get_value("marta", broken))
        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_error_leaves_no_unawaited_coroutines(self):
        async def broken(value):
            raise KeyError("boom")

        def never_called(value):  # pragma: no cover
            raise AssertionError("must not be called")

        loader = AsyncItemLoader()
        with pytest.raises(ValueError, match="Error with processor"):
            run(loader.get_value("marta", broken, never_called))
        # gc.collect() in run() plus the module-level RuntimeWarning error
        # filter make this test fail if any coroutine was left un-awaited.


class TestSyncApiUnchanged:
    def test_sync_loader_does_not_require_an_event_loop(self):
        loader = ItemLoader()
        loader.add_value("name", "marta")
        assert loader.load_item() == {"name": ["marta"]}
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()

    def test_async_loader_keeps_sync_helpers(self):
        loader = AsyncItemLoader()
        run(loader.add_value("name", "marta"))
        assert loader.get_collected_values("name") == ["marta"]
        assert isinstance(loader.get_input_processor("name"), Identity)
        assert isinstance(loader.get_output_processor("name"), Identity)

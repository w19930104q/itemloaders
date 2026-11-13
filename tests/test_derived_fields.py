"""Tests for :class:`itemloaders.DerivedField` and derived field handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from parsel import Selector

from itemloaders import DerivedField, ItemLoader
from itemloaders.processors import MapCompose, TakeFirst

# --------------------------------------------------------------------------- #
# Basic computation
# --------------------------------------------------------------------------- #


class ProductLoader(ItemLoader):
    default_output_processor = TakeFirst()

    unit_price_in = MapCompose(float)
    quantity_in = MapCompose(int)
    tax_rate_in = MapCompose(float)

    derived_fields = {
        "net_total": DerivedField(
            lambda unit_price, quantity: unit_price * quantity,
            ["unit_price", "quantity"],
        ),
        "gross_total": DerivedField(
            lambda net_total, tax_rate: round(net_total * (1 + tax_rate), 2),
            ["net_total", "tax_rate"],
        ),
    }


class TestDerivedFieldBasics:
    def test_callback_receives_final_output_values(self):
        loader = ProductLoader()
        loader.add_value("unit_price", "9.99")
        loader.add_value("quantity", "3")
        loader.add_value("tax_rate", "0.2")

        assert loader.load_item() == {
            "unit_price": 9.99,
            "quantity": 3,
            "tax_rate": 0.2,
            "net_total": 29.97,
            "gross_total": 35.96,
        }

    def test_string_depends_on_is_normalized(self):
        class SingleDepLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"shout": DerivedField(lambda name: name.upper(), "name")}

        loader = SingleDepLoader()
        loader.add_value("name", "marta")
        assert loader.load_item() == {"name": "marta", "shout": "MARTA"}

    def test_derived_value_goes_through_input_and_output_processors(self):
        class ProcessorLoader(ItemLoader):
            # Input processor upper-cases each element of the callback result.
            label_in = MapCompose(str.upper)
            label_out = TakeFirst()
            derived_fields = {
                "label": DerivedField(lambda name: name, "name"),
            }

        loader = ProcessorLoader()
        loader.add_value("name", "tv")
        # The dependency's output is ["tv"], the input processor upper-cases
        # that single element, then TakeFirst unwraps the resulting list.
        assert loader.load_item() == {"name": ["tv"], "label": "TV"}

    def test_callback_arguments_follow_dependency_order(self):
        seen = []

        def callback(first: Any, second: Any) -> Any:
            seen.extend([first, second])
            return "ok"

        class OrderLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "result": DerivedField(callback, ["second", "first"]),
            }

        loader = OrderLoader()
        loader.add_value("first", 1)
        loader.add_value("second", 2)
        loader.load_item()
        # The dependency order, not collection order, decides the arguments.
        assert seen == [2, 1]

    def test_multi_level_dependency_chain(self):
        class ChainLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "a": DerivedField(lambda x: x + 1, "x"),
                "b": DerivedField(lambda a: a * 2, "a"),
                "c": DerivedField(lambda b: b - 3, "b"),
            }

        loader = ChainLoader()
        loader.add_value("x", 10)
        assert loader.load_item() == {"x": 10, "a": 11, "b": 22, "c": 19}

    def test_each_field_is_evaluated_only_once_per_load(self):
        calls: dict[str, int] = {}

        def counting(name: str) -> Any:
            def inner(*values: Any) -> Any:
                calls[name] = calls.get(name, 0) + 1
                return values[0]

            return inner

        class OnceLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "base": DerivedField(counting("base"), "x"),
                "left": DerivedField(counting("left"), "base"),
                "right": DerivedField(counting("right"), "base"),
                "top": DerivedField(counting("top"), ["left", "right"]),
            }

        loader = OnceLoader()
        loader.add_value("x", 5)
        loader.load_item()
        assert calls == {"base": 1, "left": 1, "right": 1, "top": 1}

    def test_get_output_value_does_not_write_the_item(self):
        loader = ProductLoader()
        loader.add_value("unit_price", "10")
        loader.add_value("quantity", "2")
        loader.add_value("tax_rate", "0.1")

        assert loader.get_output_value("gross_total") == 22.0
        # No derived (or other) value leaked into the item.
        assert dict(loader.item) == {}

    def test_get_output_value_computes_only_required_subgraph(self):
        calls: set[str] = set()

        def tracker(name: str) -> Any:
            def inner(value: Any) -> Any:
                calls.add(name)
                return value

            return inner

        class SubgraphLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "used": DerivedField(tracker("used"), "x"),
                "unused": DerivedField(tracker("unused"), "y"),
            }

        loader = SubgraphLoader()
        loader.add_value("x", 1)
        loader.add_value("y", 2)
        assert loader.get_output_value("used") == 1
        assert calls == {"used"}

    def test_get_output_value_of_plain_field_is_unchanged(self):
        loader = ItemLoader()
        loader.add_value("plain", "value")
        assert loader.get_output_value("plain") == ["value"]
        assert loader.get_output_value("never_collected") == []

    def test_derived_from_initial_item_values(self):
        class InitialLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "sum": DerivedField(lambda a, b: a + b, ["a", "b"]),
            }

        # Values of a passed item are kept as-is and feed the callback through
        # their output processor, just like collected values.
        loader = InitialLoader(item={"a": 2, "b": 3})
        assert loader.load_item() == {"a": 2, "b": 3, "sum": 5}

    def test_zero_dependency_derived_field(self):
        class ConstantLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"constant": DerivedField(lambda: 42)}

        assert ConstantLoader().load_item() == {"constant": 42}
        assert ConstantLoader().get_output_value("constant") == 42


# --------------------------------------------------------------------------- #
# Diamond dependency
# --------------------------------------------------------------------------- #


class TestDiamondDependency:
    def test_shared_dependency_is_evaluated_once(self):
        base_calls = 0

        def base_callback(x: Any) -> Any:
            nonlocal base_calls
            base_calls += 1
            return x * 10

        class DiamondLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "base": DerivedField(base_callback, "x"),
                "left": DerivedField(lambda base: base + 1, "base"),
                "right": DerivedField(lambda base: base + 2, "base"),
                "top": DerivedField(
                    lambda left, right: left + right, ["left", "right"]
                ),
            }

        loader = DiamondLoader()
        loader.add_value("x", 5)
        item = loader.load_item()
        assert item == {"x": 5, "base": 50, "left": 51, "right": 52, "top": 103}
        assert base_calls == 1

    def test_shared_dependency_memoized_in_get_output_value(self):
        base_calls = 0

        def base_callback(x: Any) -> Any:
            nonlocal base_calls
            base_calls += 1
            return x

        class DiamondLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "base": DerivedField(base_callback, "x"),
                "left": DerivedField(lambda base: base, "base"),
                "right": DerivedField(lambda base: base, "base"),
                "top": DerivedField(
                    lambda left, right: left + right, ["left", "right"]
                ),
            }

        loader = DiamondLoader()
        loader.add_value("x", 7)
        assert loader.get_output_value("top") == 14
        assert base_calls == 1


# --------------------------------------------------------------------------- #
# Re-evaluation after add / replace and failure retry
# --------------------------------------------------------------------------- #


class TestReevaluation:
    def test_append_then_reload_recomputes(self):
        loader = ProductLoader()
        loader.add_value("unit_price", "10")
        loader.add_value("quantity", "1")
        loader.add_value("tax_rate", "0")
        first = loader.load_item()
        assert first["net_total"] == 10.0

        loader.add_value("quantity", "2")  # collected quantities are now [1, 2]
        second = loader.load_item()
        # TakeFirst keeps the first quantity, but the memo must be rebuilt.
        assert second["net_total"] == 10.0
        assert loader.get_collected_values("quantity") == [1, 2]

    def test_replace_then_reload_recomputes(self):
        loader = ProductLoader()
        loader.add_value("unit_price", "10")
        loader.add_value("quantity", "1")
        loader.add_value("tax_rate", "0")
        assert loader.load_item()["net_total"] == 10.0

        loader.replace_value("quantity", "4")
        assert loader.load_item()["net_total"] == 40.0

    def test_failed_load_can_be_retried_after_fixing_input(self):
        attempts = {"n": 0}

        def fragile(value: Any) -> Any:
            attempts["n"] += 1
            if value < 0:
                msg = "negative value"
                raise ValueError(msg)
            return value * 2

        class RetryLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"doubled": DerivedField(fragile, "v")}

        loader = RetryLoader()
        loader.add_value("v", -1)
        with pytest.raises(ValueError, match="derived field callback"):
            loader.load_item()
        # The failed load must not leave a half-filled item.
        assert dict(loader.item) == {}

        loader.replace_value("v", 3)
        assert loader.load_item() == {"v": 3, "doubled": 6}
        assert attempts["n"] == 2

    def test_failed_output_processor_leaves_item_untouched(self):
        class BoomLoader(ItemLoader):
            bad_out = MapCompose(int)
            default_output_processor = TakeFirst()
            derived_fields = {"bad": DerivedField(lambda: "not-a-number")}

        loader = BoomLoader()
        with pytest.raises(ValueError, match="output processor"):
            loader.load_item()
        assert dict(loader.item) == {}


# --------------------------------------------------------------------------- #
# Empty and None values
# --------------------------------------------------------------------------- #


@dataclass
class EmptyProduct:
    name: str | None = None
    shout: str | None = None
    label: str | None = None


class TestEmptyValues:
    def test_missing_schema_field_is_empty(self):
        class EmptyLoader(ItemLoader):
            default_item_class = EmptyProduct
            default_output_processor = TakeFirst()
            derived_fields = {
                "shout": DerivedField(lambda name: (name or "").upper(), "name"),
            }

        loader = EmptyLoader()
        assert loader.get_output_value("shout") is None
        item = loader.load_item()
        assert item == EmptyProduct(name=None, shout=None, label=None)

    def test_empty_value_propagates_through_chain(self):
        class EmptyLoader(ItemLoader):
            default_item_class = EmptyProduct
            default_output_processor = TakeFirst()
            derived_fields = {
                "shout": DerivedField(lambda name: (name or "").upper(), "name"),
                "label": DerivedField(
                    lambda shout: f"[{shout}]" if shout else "[]", "shout"
                ),
            }

        loader = EmptyLoader()
        assert loader.get_output_value("label") == "[]"

    def test_filtered_out_dependency_is_empty_on_dict(self):
        class FilteredLoader(ItemLoader):
            derived_fields = {
                "d": DerivedField(lambda value: value or "EMPTY", "x"),
            }

        loader = FilteredLoader()
        loader.add_value("x", None)  # None is never collected
        assert loader.get_output_value("d") == ["EMPTY"]

    def test_callback_returning_none_is_not_written(self):
        class NoneLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"d": DerivedField(lambda: None)}

        # None goes through the input processor (dropped) and TakeFirst on an
        # empty collected list returns None.
        loader = NoneLoader()
        assert loader.get_output_value("d") is None
        assert loader.load_item() == {}

    def test_identity_output_keeps_empty_list_value(self):
        # With the default Identity output processor, an all-None callback
        # result becomes an empty list, which is a value and is written.
        class EmptyListLoader(ItemLoader):
            derived_fields = {"d": DerivedField(lambda: None)}

        assert EmptyListLoader().load_item() == {"d": []}

    def test_no_derived_declarations_behaves_as_before(self):
        loader = ItemLoader()
        loader.add_value("name", "marta")
        assert loader.load_item() == {"name": ["marta"]}


# --------------------------------------------------------------------------- #
# Inheritance and override
# --------------------------------------------------------------------------- #


class BaseDerivedLoader(ItemLoader):
    default_output_processor = TakeFirst()
    derived_fields = {
        "total": DerivedField(lambda a, b: a + b, ["a", "b"]),
        "tag": DerivedField(lambda: "base"),
    }


class ChildDerivedLoader(BaseDerivedLoader):
    derived_fields = {
        **BaseDerivedLoader.derived_fields,
        "tag": DerivedField(lambda: "child"),
        "doubled": DerivedField(lambda total: total * 2, "total"),
    }


class TestInheritance:
    def test_inherited_declaration(self):
        loader = BaseDerivedLoader()
        loader.add_value("a", 1)
        loader.add_value("b", 2)
        assert loader.load_item() == {"a": 1, "b": 2, "total": 3, "tag": "base"}

    def test_override_and_new_declaration(self):
        loader = ChildDerivedLoader()
        loader.add_value("a", 1)
        loader.add_value("b", 2)
        assert loader.load_item() == {
            "a": 1,
            "b": 2,
            "total": 3,
            "tag": "child",
            "doubled": 6,
        }

    def test_grandchild_overrides_callback(self):
        class GrandChild(ChildDerivedLoader):
            derived_fields = {
                **ChildDerivedLoader.derived_fields,
                "tag": DerivedField(lambda: "grandchild"),
            }

        loader = GrandChild()
        loader.add_value("a", 0)
        loader.add_value("b", 0)
        assert loader.load_item()["tag"] == "grandchild"

    def test_override_dependency_set(self):
        class RetargetLoader(BaseDerivedLoader):
            derived_fields = {
                **BaseDerivedLoader.derived_fields,
                "total": DerivedField(lambda a, b: a - b, ["a", "b"]),
            }

        loader = RetargetLoader()
        loader.add_value("a", 5)
        loader.add_value("b", 2)
        assert loader.load_item()["total"] == 3


# --------------------------------------------------------------------------- #
# Validation: cycles and unknown dependencies, before any processor runs
# --------------------------------------------------------------------------- #


class TestGraphValidation:
    def test_self_cycle_reports_path(self):
        class SelfCycleLoader(ItemLoader):
            derived_fields = {"a": DerivedField(lambda a: a, "a")}

        loader = SelfCycleLoader()
        with pytest.raises(
            ValueError, match=r"Cyclic derived field dependency detected: a -> a"
        ):
            loader.get_output_value("a")
        with pytest.raises(ValueError, match="a -> a"):
            loader.load_item()

    def test_long_cycle_reports_full_path(self):
        class CycleLoader(ItemLoader):
            derived_fields = {
                "a": DerivedField(lambda b: b, "b"),
                "b": DerivedField(lambda c: c, "c"),
                "c": DerivedField(lambda a: a, "a"),
            }

        # The message names the field where the cycle closes and the full
        # dependency path leading back to it. Declaration order makes the
        # starting node deterministic.
        with pytest.raises(
            ValueError,
            match=r"Cyclic derived field dependency detected: a -> b -> c -> a",
        ):
            CycleLoader().load_item()

    def test_unknown_dependency_reports_path_on_typed_item(self):
        class UnknownLoader(ItemLoader):
            default_item_class = EmptyProduct
            derived_fields = {
                "shout": DerivedField(lambda name: name, "ghost"),
            }

        with pytest.raises(
            ValueError,
            match=r"Unknown derived field dependency: shout -> ghost",
        ):
            UnknownLoader().load_item()

    def test_nested_unknown_dependency_path(self):
        class NestedUnknownLoader(ItemLoader):
            default_item_class = EmptyProduct
            derived_fields = {
                "outer": DerivedField(lambda inner: inner, "inner"),
                "inner": DerivedField(lambda ghost: ghost, "ghost"),
            }

        with pytest.raises(
            ValueError,
            match=r"Unknown derived field dependency: outer -> inner -> ghost",
        ):
            NestedUnknownLoader().load_item()

    def test_validation_runs_before_processors(self):
        ran = {"output": False, "callback": False}

        class BoomOutput:
            def __call__(self, value: Any) -> Any:
                ran["output"] = True
                return value

        def boom_callback(value: Any) -> Any:
            ran["callback"] = True
            return value

        class InvalidLoader(ItemLoader):
            default_item_class = EmptyProduct
            x_out = BoomOutput()
            derived_fields = {
                "a": DerivedField(boom_callback, "missing"),
            }

        loader = InvalidLoader()
        loader.add_value("name", "tv")
        with pytest.raises(ValueError, match="Unknown derived field dependency"):
            loader.load_item()
        # Graph validation failed before any output processor or callback ran.
        assert ran == {"output": False, "callback": False}

    def test_unknown_dependency_allowed_on_schema_less_dict_when_uncollected(self):
        # A plain dict accepts any field name; an uncollected one resolves to
        # the empty value instead of being rejected.
        class OpenLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {
                "d": DerivedField(lambda value: value or "empty", "ghost"),
            }

        assert OpenLoader().get_output_value("d") == "empty"


# --------------------------------------------------------------------------- #
# Explicit collection of a derived field name is rejected
# --------------------------------------------------------------------------- #


class TestExplicitCollectionRejected:
    @pytest.mark.parametrize("method", ["add_value", "replace_value"])
    def test_direct_value_methods_raise(self, method):
        class GuardedLoader(ItemLoader):
            derived_fields = {"d": DerivedField(lambda: 1)}

        loader = GuardedLoader()
        with pytest.raises(ValueError, match="'d' is a derived field"):
            getattr(loader, method)("d", 1)

    @pytest.mark.parametrize("method", ["add_value", "replace_value"])
    def test_multi_field_dict_path_raises(self, method):
        class GuardedLoader(ItemLoader):
            derived_fields = {"d": DerivedField(lambda: 1)}

        loader = GuardedLoader()
        with pytest.raises(ValueError, match="'d' is a derived field"):
            getattr(loader, method)(None, {"d": 1, "other": 2})

    def test_xpath_and_css_collection_raise(self):
        selector = Selector(text="<html><body><p>x</p></body></html>")

        class GuardedLoader(ItemLoader):
            derived_fields = {"d": DerivedField(lambda: 1)}

        loader = GuardedLoader(selector=selector)
        with pytest.raises(ValueError, match="'d' is a derived field"):
            loader.add_xpath("d", "//p/text()")
        with pytest.raises(ValueError, match="'d' is a derived field"):
            loader.replace_css("d", "p::text")

    def test_initial_item_with_derived_field_name_raises(self):
        class GuardedLoader(ItemLoader):
            derived_fields = {"d": DerivedField(lambda: 1)}

        with pytest.raises(ValueError, match="'d' is a derived field"):
            GuardedLoader(item={"d": 1})


# --------------------------------------------------------------------------- #
# Loader context
# --------------------------------------------------------------------------- #


def scale_callback(value: Any, loader_context: Any) -> Any:
    return value * loader_context["factor"]


class TestLoaderContext:
    def test_callback_receives_context(self):
        class ContextLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"scaled": DerivedField(scale_callback, "v")}

        loader = ContextLoader(factor=3)
        loader.add_value("v", 4)
        assert loader.load_item()["scaled"] == 12

    def test_context_changes_are_picked_up_on_reload(self):
        class ContextLoader(ItemLoader):
            default_output_processor = TakeFirst()
            derived_fields = {"scaled": DerivedField(scale_callback, "v")}

        loader = ContextLoader(factor=2)
        loader.add_value("v", 4)
        assert loader.load_item()["scaled"] == 8
        loader.context["factor"] = 5
        assert loader.load_item()["scaled"] == 20


# --------------------------------------------------------------------------- #
# Nested loaders
# --------------------------------------------------------------------------- #


PRODUCT_HTML = """
<html>
  <body>
    <span class="unit_price">9.99</span>
    <span class="quantity">3</span>
    <span class="tax_rate">0.2</span>
  </body>
</html>
"""


class TestNestedLoaders:
    def test_nested_loader_shares_definitions_and_collected_values(self):
        root = ProductLoader(selector=Selector(text=PRODUCT_HTML))
        nested = root.nested_xpath("//body")
        nested.add_xpath("unit_price", 'span[@class="unit_price"]/text()')
        nested.add_xpath("quantity", 'span[@class="quantity"]/text()')
        nested.add_xpath("tax_rate", 'span[@class="tax_rate"]/text()')

        # Both the nested and the root loader resolve the same derived fields.
        assert nested.get_output_value("gross_total") == 35.96
        assert root.get_output_value("gross_total") == 35.96
        assert root.load_item() == {
            "unit_price": 9.99,
            "quantity": 3,
            "tax_rate": 0.2,
            "net_total": 29.97,
            "gross_total": 35.96,
        }

    def test_nested_loader_uses_its_own_context(self):
        class NestedContextLoader(ItemLoader):
            default_output_processor = TakeFirst()
            v_in = MapCompose(int)
            derived_fields = {"scaled": DerivedField(scale_callback, "v")}

        root = NestedContextLoader(
            selector=Selector(text="<div><a>3</a></div>"), factor=10
        )
        child = root.nested_xpath("//div", factor=100)
        child.add_xpath("v", "a/text()")

        assert child.get_output_value("scaled") == 300
        assert root.get_output_value("scaled") == 30

    def test_nested_loader_rejects_explicit_collection(self):
        root = ProductLoader(selector=Selector(text=PRODUCT_HTML))
        nested = root.nested_xpath("//body")
        with pytest.raises(ValueError, match="'net_total' is a derived field"):
            nested.add_value("net_total", 1)


# --------------------------------------------------------------------------- #
# HTML and JSON end-to-end regressions
# --------------------------------------------------------------------------- #


class TestHtmlRegression:
    def test_full_product_page(self):
        loader = ProductLoader(selector=Selector(text=PRODUCT_HTML))
        loader.add_xpath("unit_price", '//span[@class="unit_price"]/text()')
        loader.add_xpath("quantity", '//span[@class="quantity"]/text()')
        loader.add_xpath("tax_rate", '//span[@class="tax_rate"]/text()')
        assert loader.load_item() == {
            "unit_price": 9.99,
            "quantity": 3,
            "tax_rate": 0.2,
            "net_total": 29.97,
            "gross_total": 35.96,
        }


PRODUCT_JSON = """
{
  "unit_price": "12.5",
  "quantity": "4",
  "tax_rate": "0.1"
}
"""


class TestJsonRegression:
    def test_full_json_payload(self):
        selector = Selector(text=PRODUCT_JSON)
        if not hasattr(selector, "jmespath"):
            pytest.skip("parsel < 1.8.1 has no JMESPath support")

        loader = ProductLoader(selector=selector)
        loader.add_jmes("unit_price", "unit_price")
        loader.add_jmes("quantity", "quantity")
        loader.add_jmes("tax_rate", "tax_rate")
        assert loader.load_item() == {
            "unit_price": 12.5,
            "quantity": 4,
            "tax_rate": 0.1,
            "net_total": 50.0,
            "gross_total": 55.0,
        }

    def test_json_via_add_value_dict(self):
        payload = json.loads(PRODUCT_JSON)
        loader = ProductLoader()
        loader.add_value(None, payload)
        assert loader.load_item() == {
            "unit_price": 12.5,
            "quantity": 4,
            "tax_rate": 0.1,
            "net_total": 50.0,
            "gross_total": 55.0,
        }


# --------------------------------------------------------------------------- #
# Supported item types
# --------------------------------------------------------------------------- #


@dataclass
class DataclassProduct:
    unit_price: float | None = None
    quantity: int | None = None
    total: float | None = None


class TestItemTypes:
    def test_dict_item(self):
        loader = ProductLoader()
        loader.add_value("unit_price", "2.5")
        loader.add_value("quantity", "4")
        loader.add_value("tax_rate", "0")
        item = loader.load_item()
        assert isinstance(item, dict)
        assert item["net_total"] == 10.0

    def test_dataclass_item(self):
        class DLoader(ItemLoader):
            default_item_class = DataclassProduct
            default_output_processor = TakeFirst()
            unit_price_in = MapCompose(float)
            quantity_in = MapCompose(int)
            derived_fields = {
                "total": DerivedField(
                    lambda unit_price, quantity: unit_price * quantity,
                    ["unit_price", "quantity"],
                ),
            }

        loader = DLoader()
        loader.add_value("unit_price", "2.5")
        loader.add_value("quantity", "4")
        assert loader.load_item() == DataclassProduct(
            unit_price=2.5, quantity=4, total=10.0
        )

    def test_scrapy_item(self):
        scrapy = pytest.importorskip("scrapy")

        class SItem(scrapy.Item):
            unit_price = scrapy.Field()
            quantity = scrapy.Field()
            total = scrapy.Field()

        class SLoader(ItemLoader):
            default_item_class = SItem
            default_output_processor = TakeFirst()
            unit_price_in = MapCompose(float)
            quantity_in = MapCompose(int)
            derived_fields = {
                "total": DerivedField(
                    lambda unit_price, quantity: unit_price * quantity,
                    ["unit_price", "quantity"],
                ),
            }

        loader = SLoader()
        loader.add_value("unit_price", "2.5")
        loader.add_value("quantity", "4")
        item = loader.load_item()
        assert isinstance(item, SItem)
        assert dict(item) == {
            "unit_price": 2.5,
            "quantity": 4,
            "total": 10.0,
        }

    def test_attrs_item(self):
        attr = pytest.importorskip("attr")

        @attr.s
        class AProduct:
            unit_price = attr.ib(default=None)
            quantity = attr.ib(default=None)
            total = attr.ib(default=None)

        class ALoader(ItemLoader):
            default_item_class = AProduct
            default_output_processor = TakeFirst()
            unit_price_in = MapCompose(float)
            quantity_in = MapCompose(int)
            derived_fields = {
                "total": DerivedField(
                    lambda unit_price, quantity: unit_price * quantity,
                    ["unit_price", "quantity"],
                ),
            }

        loader = ALoader()
        loader.add_value("unit_price", "2.5")
        loader.add_value("quantity", "4")
        assert loader.load_item() == AProduct(unit_price=2.5, quantity=4, total=10.0)

    def test_derived_field_absent_when_none_on_dataclass_with_default(self):
        @dataclass
        class Partial:
            name: str | None = None
            missing: str | None = None

        class PartialLoader(ItemLoader):
            default_item_class = Partial
            default_output_processor = TakeFirst()
            derived_fields = {
                "missing": DerivedField(lambda: None),
            }

        loader = PartialLoader()
        loader.add_value("name", "tv")
        # The None derived value is not assigned, leaving the field default.
        assert loader.load_item() == Partial(name="tv", missing=None)

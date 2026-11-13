.. currentmodule:: itemloaders

.. _derived-fields:

Derived fields
==============

Some item fields are not extracted from the page but computed from other,
already parsed fields. For example, the tax-inclusive total price of a product
depends on the cleaned unit price, the quantity and the tax rate. Computing
such a field by hand is fragile: the computation must run after every
dependency has been collected, and calling the output processors in the right
order from a regular output processor is easy to get wrong.

A *derived field* declares the computation once, as a callback that receives
the :ref:`final output values <processors>` of the fields it depends on.

Declaring a derived field
-------------------------

Derived fields are declared in the :attr:`ItemLoader.derived_fields` class
attribute, which maps field names to :class:`DerivedField` objects::

    from itemloaders import DerivedField, ItemLoader
    from itemloaders.processors import MapCompose, TakeFirst

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

    loader = ProductLoader()
    loader.add_value("unit_price", "9.99")
    loader.add_value("quantity", "3")
    loader.add_value("tax_rate", "0.2")
    loader.load_item()
    # {'unit_price': 9.99, 'quantity': 3, 'tax_rate': 0.2,
    #  'net_total': 29.97, 'gross_total': 35.96}

The callback of ``gross_total`` receives the output value of ``net_total``,
itself a derived field: derived fields may depend on each other, in chains or
in :ref:`diamond shapes <derived-fields-diamonds>`.

The second argument of :class:`DerivedField`, ``depends_on``, lists the
dependency field names in the order in which their output values are passed as
positional arguments to the callback. A single field name may be given
directly instead of a sequence.

Processing of the callback result
---------------------------------

The value returned by the callback is not written to the item as-is. It goes
through the input processor and then the output processor declared for the
derived field, exactly like a value collected with
:meth:`~ItemLoader.add_value`::

    class LabelLoader(ItemLoader):
        label_in = MapCompose(str.upper)
        label_out = TakeFirst()
        derived_fields = {
            "label": DerivedField(lambda name: name, "name"),
        }

    loader = LabelLoader()
    loader.add_value("name", "tv")
    loader.load_item()
    # {'name': ['tv'], 'label': 'TV'}

Like collected values, a derived value is only assigned to the item when its
output processor does not return ``None``.

Dependency values
-----------------

Callbacks receive the *final output values* of their dependencies, that is the
values produced by the dependencies' output processors, not the raw collected
lists. With the default :class:`~itemloaders.processors.Identity` output
processor a dependency that was never collected resolves to an empty list;
with :class:`~itemloaders.processors.TakeFirst` it resolves to ``None``. This
lets a callback decide what an empty dependency means::

    class ShoutLoader(ItemLoader):
        default_output_processor = TakeFirst()
        derived_fields = {
            "shout": DerivedField(
                lambda name: (name or "").upper(), "name"
            ),
        }

The callback may also accept the active :ref:`loader context
<loaders-context>` through a ``loader_context`` keyword argument, just like
input and output processors::

    def scale(value, loader_context):
        return value * loader_context["factor"]

    class ScaledLoader(ItemLoader):
        default_output_processor = TakeFirst()
        derived_fields = {"scaled": DerivedField(scale, "value")}

    loader = ScaledLoader(factor=3)
    loader.add_value("value", 4)
    loader.load_item()["scaled"]
    # 12

How and when fields are evaluated
---------------------------------

-   Within one call to :meth:`~ItemLoader.load_item` or
    :meth:`~ItemLoader.get_output_value`, every field output is computed at
    most once. The dependencies of a derived field are evaluated first, on
    demand, and their result is memoized for the rest of the call.

-   The memo does not survive the call. Collecting more data with an
    ``add_*`` method, or replacing it with a ``replace_*`` method, therefore
    recomputes every affected derived field the next time the item is loaded.

-   :meth:`~ItemLoader.get_output_value` only computes the field requested and
    the fields it depends on. It never writes to the item.

-   :meth:`~ItemLoader.load_item` computes every field output first and only
    then writes the item, so a failing processor or callback cannot leave a
    half-populated item. After fixing the collected data, the loader can be
    loaded again.

.. _derived-fields-diamonds:

Diamond dependencies
--------------------

When several derived fields share a dependency, for example::

    class DiamondLoader(ItemLoader):
        default_output_processor = TakeFirst()
        derived_fields = {
            "base": DerivedField(lambda x: x * 10, "x"),
            "left": DerivedField(lambda base: base + 1, "base"),
            "right": DerivedField(lambda base: base + 2, "base"),
            "top": DerivedField(
                lambda left, right: left + right, ["left", "right"]
            ),
        }

the shared ``base`` field is evaluated once and its output is reused by both
``left`` and ``right`` (and, transitively, by ``top``).

Invalid dependency graphs
-------------------------

The dependency graph is validated before any processor or callback runs. Two
situations raise a :exc:`ValueError` whose message contains the full
dependency path:

-   A **cycle**, including a field that depends on itself::

        Cyclic derived field dependency detected: a -> b -> c -> a

-   An **unknown dependency** on items with a fixed schema (dataclasses,
    :class:`Scrapy items <scrapy.item.Item>`, ...), i.e. a name that is neither
    a declared field of the item nor another derived field::

        Unknown derived field dependency: outer -> inner -> ghost

    Schema-less items such as plain :class:`dict` objects accept any field
    name, so an uncollected dependency there is treated as an empty dependency
    rather than an unknown one.

A derived field cannot be collected explicitly: calling any ``add_*`` or
``replace_*`` method (including the ``field_name=None`` multi-field form) with
a derived field name raises a :exc:`ValueError`. The same applies to passing
an initial item that already contains a value for a derived field name.

Inheritance
-----------

Derived field declarations follow normal class inheritance. A subclass
inherits the declarations of its parents, may add new ones and may override an
inherited declaration by reusing the field name::

    class BaseLoader(ItemLoader):
        derived_fields = {
            "total": DerivedField(lambda a, b: a + b, ["a", "b"]),
            "tag": DerivedField(lambda: "base"),
        }

    class SiteLoader(BaseLoader):
        derived_fields = {
            **BaseLoader.derived_fields,
            "tag": DerivedField(lambda: "site"),
            "doubled": DerivedField(lambda total: total * 2, "total"),
        }

Nested loaders
--------------

:ref:`Nested loaders <nested-loaders>` share the same item and the same
derived field declarations as their parent. A derived field can be resolved
from any loader of the hierarchy, using that loader's
:ref:`context <loaders-context>`. Values collected through a nested loader are
visible to the derived fields resolved from the root loader and vice versa.

Loaders without derived fields
------------------------------

A loader that does not declare :attr:`~ItemLoader.derived_fields` keeps the
exact behavior of previous versions: no dependency validation is performed and
the output processors run in the usual way.

.. versionadded:: VERSION

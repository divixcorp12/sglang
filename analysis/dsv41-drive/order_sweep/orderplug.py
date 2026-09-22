"""Reorders collected tests: ORDER=alpha (unittest's order: classes and methods by name), file (pytest's),
rev (reverse alphabetical classes, methods by name). Only reorders; selects nothing."""
import os


def pytest_collection_modifyitems(session, config, items):
    order = os.environ.get("ORDER", "file")
    if order == "file":
        return

    def key(item):
        cls = item.cls.__name__ if item.cls is not None else ""
        return (cls, item.name)

    if order == "alpha":
        items.sort(key=key)
    elif order == "rev":
        # classes in reverse alphabetical order, methods within a class alphabetical (unittest's within-class order)
        items.sort(key=lambda i: ((i.cls.__name__ if i.cls is not None else ""), i.name))
        groups = {}
        for i in items:
            groups.setdefault(i.cls.__name__ if i.cls is not None else "", []).append(i)
        items[:] = [i for name in sorted(groups, reverse=True) for i in groups[name]]

import asyncio

import pytest

from hyperion.context import RequestContext, context_scope, current_context


def test_context_scope_overlays_fields_and_restores_nested_context():
    assert current_context() == RequestContext()

    with context_scope(request_id="req-1", session_id="sess-1"):
        assert current_context() == RequestContext(request_id="req-1", session_id="sess-1")

        with context_scope(call_site="router", request_id=""):
            assert current_context() == RequestContext(
                request_id="",
                session_id="sess-1",
                call_site="router",
            )

        assert current_context() == RequestContext(request_id="req-1", session_id="sess-1")

    assert current_context() == RequestContext()


def test_context_scope_restores_exact_prior_context_after_exception():
    outer = RequestContext(request_id="req-outer")

    with context_scope(request_id=outer.request_id):
        try:
            with context_scope(request_id="req-inner", session_id="sess-inner"):
                assert current_context().request_id == "req-inner"
                raise RuntimeError("boom")
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("context scope did not raise")

        assert current_context() == outer

    assert current_context() == RequestContext()


def test_context_scope_treats_omitted_values_as_inherited_and_none_as_clear():
    with context_scope(request_id="req-1", session_id="sess-1", call_site="agent"), context_scope(
        session_id=None,
    ):
        assert current_context() == RequestContext(
            request_id="req-1",
            session_id=None,
            call_site="agent",
        )


def test_request_context_metadata_is_immutable_and_overlay_replaces_keys():
    metadata = {"tenant": "a"}
    base = RequestContext(metadata=metadata)
    metadata["tenant"] = "changed"
    derived = base.overlay(metadata={"tenant": "b", "trace": "1"})

    assert base.metadata == {"tenant": "a"}
    assert derived.metadata == {"tenant": "b", "trace": "1"}

    with pytest.raises(TypeError):
        derived.metadata["tenant"] = "changed"


def test_request_context_direct_construction_validates_declared_types():
    with pytest.raises(TypeError, match="request_id"):
        RequestContext(request_id=123)
    with pytest.raises(TypeError, match="session_id"):
        RequestContext(session_id=object())
    with pytest.raises(TypeError, match="call_site"):
        RequestContext(call_site=False)
    with pytest.raises(TypeError, match="metadata must be a mapping or None"):
        RequestContext(metadata=[("tenant", "a")])
    with pytest.raises(TypeError, match="metadata keys must be str"):
        RequestContext(metadata={1: "a"})
    with pytest.raises(TypeError, match="metadata values must be str"):
        RequestContext(metadata={"tenant": 1})


def test_context_var_isolated_between_async_tasks():
    async def observe(request_id: str) -> tuple[str | None, str | None]:
        with context_scope(request_id=request_id):
            await asyncio.sleep(0)
            inside = current_context().request_id
        return inside, current_context().request_id

    async def main() -> list[tuple[str | None, str | None]]:
        return await asyncio.gather(observe("req-a"), observe("req-b"))

    assert asyncio.run(main()) == [
        ("req-a", None),
        ("req-b", None),
    ]

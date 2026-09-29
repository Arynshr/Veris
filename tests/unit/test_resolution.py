import pytest
from conftest import FakeSearchProvider

from veris.core.models import QueryUnderstanding, SubjectMode
from veris.intake.resolution import _is_singular_role, resolve_subjects


def test_is_singular_role():
    for role in ["CEO", "CFO", "ceo", "President", "Founder", "Co-Founder", "chairman"]:
        assert _is_singular_role(role), role
    for role in [
        "board of directors",
        "executives",
        "leadership team",
        "management team",
        "c-suite",
    ]:
        assert not _is_singular_role(role), role


@pytest.mark.asyncio
async def test_ceo_and_cfo_resolves_to_exactly_two_people(settings_factory):
    """The headline regression test: 'CEO and CFO of Acme' must resolve to
    exactly 2 people, never the whole board/article cast of names that used
    to leak in from blind whole-snippet entity extraction."""
    understanding = QueryUnderstanding(
        mode=SubjectMode.ROLE_GROUP,
        organization="Acme",
        role_terms=["CEO", "CFO"],
        explicit_subjects=[],
        raw_query="the CEO and CFO of Acme",
    )
    responses = {
        "who is the CEO of Acme": [
            (
                "Acme names Jane Doe as CEO",
                "Jane Doe was appointed CEO of Acme in 2023, succeeding John Carter.",
                "https://news.example/a",
            ),
            (
                "Acme board welcomes new members",
                "Board member Alice Nguyen joined alongside Jane Doe at the CEO's request.",
                "https://news.example/b",
            ),
            (
                "Analyst notes on Acme",
                "Analyst Bob Kim covers Acme for a major bank.",
                "https://news.example/c",
            ),
        ],
        "who is the CFO of Acme": [
            (
                "Acme appoints CFO",
                "Mark Reyes was named CFO of Acme, replacing the retiring finance chief.",
                "https://news.example/d",
            ),
            (
                "Acme earnings call recap",
                "CFO Mark Reyes discussed guidance while CEO Jane Doe fielded questions.",
                "https://news.example/e",
            ),
        ],
    }
    provider = FakeSearchProvider(responses=responses)
    settings = settings_factory()

    resolved = await resolve_subjects(
        understanding, settings, provider_chain=[provider]
    )

    assert len(resolved) == 2, (
        f"expected exactly 2 resolved subjects, got {[r.name for r in resolved]}"
    )
    names = {r.name for r in resolved}
    assert names == {"Jane Doe", "Mark Reyes"}
    # people who merely co-occurred in the same articles must NOT leak in
    assert "Alice Nguyen" not in names
    assert "Bob Kim" not in names


@pytest.mark.asyncio
async def test_collective_role_capped_by_max_group_role_size(settings_factory):
    board_members = [
        "Alice Nguyen",
        "Bob Kim",
        "Carla Ruiz",
        "David Chen",
        "Emma Watson",
        "Frank Ortiz",
        "Grace Liu",
        "Henry Park",
    ]
    responses = {
        "who is the board of directors of Acme": [
            (
                "Acme board profile",
                f"Board member {name} has served on the Acme board of directors since 2019.",
                f"https://news.example/{name.replace(' ', '_')}",
            )
            for name in board_members
        ],
    }
    provider = FakeSearchProvider(responses=responses)
    settings = settings_factory(max_group_role_size=3, max_resolved_subjects=8)
    understanding = QueryUnderstanding(
        mode=SubjectMode.ROLE_GROUP,
        organization="Acme",
        role_terms=["board of directors"],
        explicit_subjects=[],
        raw_query="the board of directors of Acme",
    )

    resolved = await resolve_subjects(
        understanding, settings, provider_chain=[provider]
    )
    assert 1 <= len(resolved) <= 3


@pytest.mark.asyncio
async def test_max_resolved_subjects_overall_ceiling(settings_factory):
    """Even across multiple role terms, the overall cap must never be exceeded."""
    names = {
        "CEO": "Jane Doe",
        "CFO": "Mark Reyes",
        "COO": "Priya Shah",
        "CTO": "Sam Okafor",
        "CMO": "Lena Fischer",
    }
    responses = {
        f"who is the {role} of Acme": [
            (
                f"{role} of Acme",
                f"{name} serves as {role} of Acme.",
                f"https://news.example/{role}",
            )
        ]
        for role, name in names.items()
    }
    provider = FakeSearchProvider(responses=responses)
    settings = settings_factory(max_resolved_subjects=2)
    understanding = QueryUnderstanding(
        mode=SubjectMode.ROLE_GROUP,
        organization="Acme",
        role_terms=list(names.keys()),
        explicit_subjects=[],
        raw_query="the executive team of Acme",
    )

    resolved = await resolve_subjects(
        understanding, settings, provider_chain=[provider]
    )
    assert len(resolved) <= 2


@pytest.mark.asyncio
async def test_explicit_subjects_bypass_resolution(settings_factory):
    understanding = QueryUnderstanding(
        mode=SubjectMode.NAMED_LIST,
        explicit_subjects=["Jane Doe", "Mark Reyes"],
        raw_query="Jane Doe and Mark Reyes",
    )
    provider = FakeSearchProvider(responses={})
    resolved = await resolve_subjects(
        understanding, settings_factory(), provider_chain=[provider]
    )
    assert [r.name for r in resolved] == ["Jane Doe", "Mark Reyes"]
    assert (
        provider.calls == []
    )  # no resolution search needed - names were given directly


@pytest.mark.asyncio
async def test_single_subject_mode_passthrough(settings_factory):
    understanding = QueryUnderstanding(
        mode=SubjectMode.SINGLE,
        explicit_subjects=["solid-state batteries"],
        raw_query="solid-state batteries",
    )
    resolved = await resolve_subjects(
        understanding, settings_factory(), provider_chain=[FakeSearchProvider()]
    )
    assert [r.name for r in resolved] == ["solid-state batteries"]

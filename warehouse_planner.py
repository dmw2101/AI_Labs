"""A small breadth-first STRIPS planner for a warehouse robot."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Iterable


# Propositions are immutable strings such as "At(Robot,A)".
State = frozenset[str]


@dataclass(frozen=True)
class Action:
    """A grounded action with explicit preconditions and effects."""

    name: str
    positive_preconditions: frozenset[str] = frozenset()
    negative_preconditions: frozenset[str] = frozenset()
    positive_effects: frozenset[str] = frozenset()
    negative_effects: frozenset[str] = frozenset()

    def is_applicable(self, state: State) -> bool:
        """Return True exactly when every precondition holds in state."""
        return (
            self.positive_preconditions <= state
            and self.negative_preconditions.isdisjoint(state)
        )

    def apply(self, state: State) -> State:
        """Return the successor state produced by this action."""
        if not self.is_applicable(state):
            raise ValueError(f"Action is not applicable: {self.name}")
        return (state - self.negative_effects) | self.positive_effects


@dataclass(frozen=True)
class Goal:
    """Facts that must be true and facts that must be false."""

    positive: frozenset[str]
    negative: frozenset[str] = frozenset()

    def is_satisfied(self, state: State) -> bool:
        return self.positive <= state and self.negative.isdisjoint(state)


def breadth_first_plan(
    initial_state: State, actions: Iterable[Action], goal: Goal
) -> tuple[list[Action], list[State]] | None:
    """Find a shortest plan, returning its actions and complete state trace.

    The trace always starts with the initial state.  None means that every
    reachable state was explored without satisfying the goal.
    """
    action_list = tuple(actions)
    frontier = deque([(initial_state, [], [initial_state])])
    visited = {initial_state}

    while frontier:
        state, plan, trace = frontier.popleft()

        if goal.is_satisfied(state):
            return plan, trace

        for action in action_list:
            if not action.is_applicable(state):
                continue

            successor = action.apply(state)
            if successor in visited:
                continue

            visited.add(successor)
            frontier.append(
                (successor, plan + [action], trace + [successor])
            )

    return None


def verify_plan(
    initial_state: State, plan: Iterable[Action], goal: Goal
) -> tuple[bool, list[State]]:
    """Independently replay a plan and check every action and the final goal.

    The Boolean result is False if an action is inapplicable or if the final
    state does not satisfy the goal. The returned trace contains every state
    that was successfully reached.
    """
    state = initial_state
    trace = [state]

    for action in plan:
        if not action.is_applicable(state):
            return False, trace
        state = action.apply(state)
        trace.append(state)

    return goal.is_satisfied(state), trace


def warehouse_actions(*, include_pickup: bool = True) -> list[Action]:
    """Create grounded actions, optionally omitting every Pickup action."""
    locations = ("A", "B", "C")
    connections = (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))
    actions: list[Action] = []

    for location in locations:
        robot_here = f"At(Robot,{location})"
        package_here = f"At(Package,{location})"

        if include_pickup:
            actions.append(
                Action(
                    name=f"Pickup(Package,{location})",
                    positive_preconditions=frozenset({robot_here, package_here}),
                    negative_preconditions=frozenset({"Holding(Robot,Package)"}),
                    positive_effects=frozenset({"Holding(Robot,Package)"}),
                    negative_effects=frozenset({package_here}),
                )
            )
        actions.append(
            Action(
                name=f"Drop(Package,{location})",
                positive_preconditions=frozenset(
                    {robot_here, "Holding(Robot,Package)"}
                ),
                negative_preconditions=frozenset({package_here}),
                positive_effects=frozenset({package_here}),
                negative_effects=frozenset({"Holding(Robot,Package)"}),
            )
        )

    for source, destination in connections:
        actions.append(
            Action(
                name=f"Move({source},{destination})",
                positive_preconditions=frozenset({f"At(Robot,{source})"}),
                negative_preconditions=frozenset({f"At(Robot,{destination})"}),
                positive_effects=frozenset({f"At(Robot,{destination})"}),
                negative_effects=frozenset({f"At(Robot,{source})"}),
            )
        )

    return actions


def format_state(state: State) -> str:
    return "{" + ", ".join(sorted(state)) + "}"


def print_result(result: tuple[list[Action], list[State]] | None) -> None:
    if result is None:
        print("No plan found.")
        return

    plan, states = result
    print(f"Plan found with {len(plan)} action(s).")
    print(f"Initial state: {format_state(states[0])}")

    if not plan:
        print("The initial state already satisfies the goal.")
        return

    for step, (action, state) in enumerate(zip(plan, states[1:]), start=1):
        print(f"{step}. {action.name}")
        print(f"   State: {format_state(state)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a plan for delivering the package from A."
    )
    parser.add_argument(
        "--goal-location",
        choices=("A", "B", "C"),
        default="C",
        help="package destination (default: C)",
    )
    parser.add_argument(
        "--no-pickup",
        action="store_true",
        help="remove Pickup actions to create an impossible planning problem",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    initial_state: State = frozenset(
        {
            "At(Robot,A)",
            "At(Package,A)",
        }
    )
    goal = Goal(
        positive=frozenset({f"At(Package,{arguments.goal_location})"}),
    )

    actions = warehouse_actions(include_pickup=not arguments.no_pickup)
    result = breadth_first_plan(initial_state, actions, goal)
    print_result(result)


if __name__ == "__main__":
    main()

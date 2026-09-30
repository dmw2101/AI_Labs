"""Automated tests and recorded results for the warehouse planner."""

import unittest

from warehouse_planner import (
    Action,
    Goal,
    State,
    breadth_first_plan,
    format_state,
    verify_plan,
    warehouse_actions,
)


INITIAL_STATE: State = frozenset({"At(Robot,A)", "At(Package,A)"})
DELIVERY_GOAL = Goal(
    positive=frozenset({"At(Package,C)"}),
)


def empty_move_to_c() -> Action:
    """An irrelevant action that moves the robot, but not the package."""
    return Action(
        name="MoveEmpty(A,C)",
        positive_preconditions=frozenset({"At(Robot,A)"}),
        negative_preconditions=frozenset({"Holding(Robot,Package)"}),
        positive_effects=frozenset({"At(Robot,C)"}),
        negative_effects=frozenset({"At(Robot,A)"}),
    )


class WarehousePlannerTests(unittest.TestCase):
    def test_a_solvable_problem_and_every_action_is_valid(self) -> None:
        result = breadth_first_plan(
            INITIAL_STATE,
            warehouse_actions(),
            DELIVERY_GOAL,
        )

        self.assertIsNotNone(result)
        assert result is not None
        plan, search_trace = result

        # Independently replay every action and check its preconditions.
        is_valid, verification_trace = verify_plan(
            INITIAL_STATE, plan, DELIVERY_GOAL
        )
        self.assertTrue(is_valid)
        self.assertEqual(search_trace, verification_trace)
        self.assertEqual(
            [action.name for action in plan],
            [
                "Pickup(Package,A)",
                "Move(A,B)",
                "Move(B,C)",
                "Drop(Package,C)",
            ],
        )

    def test_b_no_plan_without_pickup_actions(self) -> None:
        result = breadth_first_plan(
            INITIAL_STATE,
            warehouse_actions(include_pickup=False),
            DELIVERY_GOAL,
        )

        self.assertIsNone(result)

    def test_c_robot_at_c_is_not_package_at_c(self) -> None:
        irrelevant_action = empty_move_to_c()

        # Prove what the irrelevant action actually does.
        robot_only_state = irrelevant_action.apply(INITIAL_STATE)
        self.assertIn("At(Robot,C)", robot_only_state)
        self.assertIn("At(Package,A)", robot_only_state)
        self.assertNotIn("At(Package,C)", robot_only_state)
        self.assertFalse(DELIVERY_GOAL.is_satisfied(robot_only_state))

        # Put the irrelevant action first so BFS examines it before useful ones.
        actions = [irrelevant_action, *warehouse_actions()]
        result = breadth_first_plan(INITIAL_STATE, actions, DELIVERY_GOAL)

        self.assertIsNotNone(result)
        assert result is not None
        plan, states = result
        is_valid, verification_trace = verify_plan(
            INITIAL_STATE, plan, DELIVERY_GOAL
        )
        self.assertTrue(is_valid)
        self.assertEqual(states, verification_trace)
        self.assertNotIn("MoveEmpty(A,C)", [action.name for action in plan])
        self.assertIn("At(Package,C)", states[-1])


def print_record(label: str, actions: list[Action]) -> None:
    """Run one scenario and print the requested test record."""
    result = breadth_first_plan(INITIAL_STATE, actions, DELIVERY_GOAL)
    print(label)
    print(f"  Initial state: {format_state(INITIAL_STATE)}")
    print("  Goal: {At(Package,C)}")

    if result is None:
        print("  Plan found: No")
        print("  Resulting plan: None")
        print("  Plan valid: N/A (the state space was exhausted)")
        return

    plan, _ = result
    is_valid, trace = verify_plan(INITIAL_STATE, plan, DELIVERY_GOAL)
    print("  Plan found: Yes")
    print("  Resulting plan:")

    state = INITIAL_STATE
    for step, action in enumerate(plan, start=1):
        applicable = action.is_applicable(state)
        state = action.apply(state)
        print(f"    {step}. {action.name} (applicable: {applicable})")
        print(f"       State: {format_state(state)}")

    print(f"  Final goal satisfied: {DELIVERY_GOAL.is_satisfied(trace[-1])}")
    print(f"  Plan valid: {is_valid}")


def print_recorded_results() -> None:
    print_record("Test A: Solvable problem", warehouse_actions())
    print()
    print_record(
        "Test B: Impossible without Pickup",
        warehouse_actions(include_pickup=False),
    )
    print()
    print_record(
        "Test C: Irrelevant robot-only action",
        [empty_move_to_c(), *warehouse_actions()],
    )


if __name__ == "__main__":
    print_recorded_results()
    print("\nAutomated assertions")
    unittest.main(argv=[__file__], verbosity=2, exit=False)

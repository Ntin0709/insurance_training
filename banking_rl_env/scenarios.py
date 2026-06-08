"""Task suite for learning banking tool selection and safe sequencing."""

from __future__ import annotations

from dataclasses import dataclass

from banking_rl_env.tools import Action, TOOLS


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    user_goal: str
    target_tool: str
    params: dict[str, object]
    intent: str
    difficulty: int = 1
    success_criteria: tuple[str, ...] = ()

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return TOOLS[self.target_tool].prerequisites

    @property
    def requires_auth(self) -> bool:
        return TOOLS[self.target_tool].requires_auth

    @property
    def requires_confirmation(self) -> bool:
        return TOOLS[self.target_tool].requires_confirmation

    def oracle_actions(self) -> tuple[str, ...]:
        actions: list[str] = []
        if self.requires_auth:
            actions.append(Action.AUTHENTICATE.value)
        actions.extend(self.prerequisites)
        if self.requires_confirmation:
            actions.append(Action.ASK_CONFIRMATION.value)
        actions.append(self.target_tool)
        actions.append(Action.FINISH.value)
        return tuple(actions)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "balance_savings",
        "Customer wants the available savings account balance.",
        "get_account_balance",
        {"customer_id": "C001", "account_type": "SAVINGS"},
        "read_private_account_data",
        1,
        ("authenticated", "balance_returned"),
    ),
    Scenario(
        "public_branch_lookup",
        "User wants branch details for Mumbai branches with lockers.",
        "get_branch_info",
        {"query": "Mumbai", "service_type": "LOCKER", "radius_km": 5},
        "public_branch_search",
        1,
        ("branch_results_returned",),
    ),
    Scenario(
        "fd_projection",
        "User wants maturity value for a 100000 INR fixed deposit for 365 days.",
        "calc_fd_maturity",
        {"principal": 100000, "interest_rate": 6.75, "tenure": 365, "payout_type": "CUMULATIVE"},
        "financial_calculation",
        1,
        ("maturity_amount_returned",),
    ),
    Scenario(
        "book_fd",
        "Customer wants to book a new fixed deposit from savings balance.",
        "book_new_fd",
        {
            "customer_id": "C001",
            "source_account_id": "SA001",
            "principal_amount": 50000,
            "tenure": 365,
            "payout_type": "CUMULATIVE",
            "idempotency_key": "00000000-0000-4000-8000-000000000001",
        },
        "state_changing_fd_booking",
        3,
        ("authenticated", "maturity_calculated", "balance_checked", "confirmed", "fd_created"),
    ),
    Scenario(
        "close_fd",
        "Customer wants to prematurely close an active FD and credit savings.",
        "request_premature_closure",
        {
            "fd_number": "FD001",
            "closure_date": "2026-06-04",
            "credit_account_id": "SA001",
            "reason": "EMERGENCY",
            "idempotency_key": "00000000-0000-4000-8000-000000000002",
        },
        "irreversible_fd_closure",
        3,
        ("authenticated", "penalty_calculated", "confirmed", "closure_submitted"),
    ),
    Scenario(
        "fraud_complaint",
        "Customer wants to report a fraudulent digital transaction.",
        "log_complaint",
        {
            "customer_id": "C001",
            "complaint_category": "FRAUD",
            "description": "Unauthorized UPI debit.",
            "account_id": "SA001",
            "severity": "HIGH",
            "preferred_contact": "CALL",
            "idempotency_key": "00000000-0000-4000-8000-000000000003",
        },
        "official_dispute_submission",
        2,
        ("authenticated", "confirmed", "complaint_logged"),
    ),
    Scenario(
        "transaction_history_upi",
        "Customer needs recent UPI debit transaction history for a dispute.",
        "get_transaction_history",
        {
            "account_id": "SA001",
            "from_date": "2026-05-01",
            "to_date": "2026-06-04",
            "transaction_type": "DEBIT",
            "limit": 10,
        },
        "read_sensitive_transactions",
        2,
        ("authenticated", "transactions_returned"),
    ),
    Scenario(
        "gold_ltv_quote",
        "User wants to know eligible loan amount for 25 grams of 22K gold.",
        "calc_gold_ltv",
        {"gold_weight_grams": 25, "purity": "22K", "city": "Mumbai"},
        "public_secured_loan_calculation",
        1,
        ("eligible_loan_returned",),
    ),
    Scenario(
        "interest_certificate",
        "Customer wants an interest certificate for financial year 2025-26 by email.",
        "generate_interest_certificate",
        {
            "customer_id": "C001",
            "financial_year": "2025-26",
            "format": "EMAIL",
            "idempotency_key": "00000000-0000-4000-8000-000000000004",
        },
        "official_document_generation",
        2,
        ("authenticated", "confirmed", "certificate_generated"),
    ),
    Scenario(
        "loan_product_home",
        "User asks about home loan product limits and documents.",
        "get_loan_product_details",
        {"product_type": "HOME", "customer_segment": "SALARIED"},
        "public_product_information",
        1,
        ("loan_product_returned",),
    ),
)

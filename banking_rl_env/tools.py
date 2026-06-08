"""Tool metadata derived from `banking_tools_reference (2).md`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AuthLevel(str, Enum):
    NONE = "none"
    AUTH = "auth"
    AUTH_CONFIRM = "auth_confirm"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    auth_level: AuthLevel
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    write_tool: bool = False

    @property
    def requires_auth(self) -> bool:
        return self.auth_level in {AuthLevel.AUTH, AuthLevel.AUTH_CONFIRM}

    @property
    def requires_confirmation(self) -> bool:
        return self.auth_level == AuthLevel.AUTH_CONFIRM


TOOLS: dict[str, ToolSpec] = {
    "get_account_balance": ToolSpec(
        "get_account_balance", "read", AuthLevel.AUTH, ("customer_id",), ("account_type",)
    ),
    "get_fd_details": ToolSpec(
        "get_fd_details", "read", AuthLevel.AUTH, ("customer_id",), ("fd_number", "status")
    ),
    "get_rd_details": ToolSpec(
        "get_rd_details", "read", AuthLevel.AUTH, ("customer_id",), ("rd_number", "status")
    ),
    "get_loan_account_details": ToolSpec(
        "get_loan_account_details",
        "read",
        AuthLevel.AUTH,
        ("customer_id",),
        ("loan_id", "include_overdue", "include_co_applicants", "include_collateral"),
    ),
    "get_transaction_history": ToolSpec(
        "get_transaction_history",
        "read",
        AuthLevel.AUTH,
        ("account_id", "from_date", "to_date"),
        ("transaction_type", "limit", "offset"),
    ),
    "get_kyc_status": ToolSpec(
        "get_kyc_status", "read", AuthLevel.AUTH, ("customer_id",), ("document_type",)
    ),
    "get_foreclosure_quote": ToolSpec(
        "get_foreclosure_quote",
        "read",
        AuthLevel.AUTH,
        ("customer_id",),
        ("loan_id", "foreclosure_date"),
    ),
    "get_branch_info": ToolSpec(
        "get_branch_info", "read", AuthLevel.NONE, ("query",), ("service_type", "radius_km")
    ),
    "get_gold_rate_today": ToolSpec(
        "get_gold_rate_today", "read", AuthLevel.NONE, (), ("purity", "city")
    ),
    "calc_fd_maturity": ToolSpec(
        "calc_fd_maturity",
        "computation",
        AuthLevel.NONE,
        ("principal", "interest_rate", "tenure"),
        ("payout_type", "compounding_frequency", "senior_citizen"),
    ),
    "calc_rd_maturity": ToolSpec(
        "calc_rd_maturity",
        "computation",
        AuthLevel.NONE,
        ("monthly_installment", "interest_rate", "tenure_months"),
        ("senior_citizen",),
    ),
    "calc_premature_penalty": ToolSpec(
        "calc_premature_penalty",
        "computation",
        AuthLevel.AUTH,
        ("fd_number", "closure_date"),
        ("reason",),
    ),
    "calc_emi": ToolSpec(
        "calc_emi",
        "computation",
        AuthLevel.NONE,
        ("principal", "interest_rate", "tenure_months"),
        ("processing_fee", "start_date"),
    ),
    "calc_gold_ltv": ToolSpec(
        "calc_gold_ltv",
        "computation",
        AuthLevel.NONE,
        ("gold_weight_grams", "purity"),
        ("ltv_percentage", "city"),
    ),
    "search_product_kb": ToolSpec(
        "search_product_kb",
        "knowledge",
        AuthLevel.NONE,
        ("query",),
        ("product_category", "top_k"),
    ),
    "search_rbi_circular": ToolSpec(
        "search_rbi_circular",
        "knowledge",
        AuthLevel.NONE,
        ("query",),
        ("from_date", "to_date", "circular_type"),
    ),
    "get_fd_rate_card": ToolSpec(
        "get_fd_rate_card",
        "knowledge",
        AuthLevel.NONE,
        (),
        ("customer_type", "tenure_range", "effective_date"),
    ),
    "get_loan_product_details": ToolSpec(
        "get_loan_product_details",
        "knowledge",
        AuthLevel.NONE,
        ("product_type",),
        ("customer_segment",),
    ),
    "generate_uuid": ToolSpec("generate_uuid", "utility", AuthLevel.NONE, (), ("count",)),
    "raise_service_request": ToolSpec(
        "raise_service_request",
        "transaction",
        AuthLevel.AUTH_CONFIRM,
        ("customer_id", "request_type", "account_id", "idempotency_key"),
        ("remarks", "priority"),
        write_tool=True,
    ),
    "book_new_fd": ToolSpec(
        "book_new_fd",
        "transaction",
        AuthLevel.AUTH_CONFIRM,
        ("customer_id", "source_account_id", "principal_amount", "tenure", "idempotency_key"),
        ("payout_type", "nominee_id", "auto_renewal", "senior_citizen"),
        ("calc_fd_maturity", "get_account_balance"),
        True,
    ),
    "request_premature_closure": ToolSpec(
        "request_premature_closure",
        "transaction",
        AuthLevel.AUTH_CONFIRM,
        ("fd_number", "closure_date", "credit_account_id", "reason", "idempotency_key"),
        ("remarks",),
        ("calc_premature_penalty",),
        True,
    ),
    "generate_interest_certificate": ToolSpec(
        "generate_interest_certificate",
        "transaction",
        AuthLevel.AUTH_CONFIRM,
        ("customer_id", "financial_year", "idempotency_key"),
        ("account_ids", "format"),
        write_tool=True,
    ),
    "log_complaint": ToolSpec(
        "log_complaint",
        "dispute",
        AuthLevel.AUTH_CONFIRM,
        ("customer_id", "complaint_category", "description", "idempotency_key"),
        ("account_id", "severity", "preferred_contact"),
        write_tool=True,
    ),
}

TOOL_NAMES = tuple(TOOLS.keys())


class Action(str, Enum):
    AUTHENTICATE = "AUTHENTICATE"
    ASK_CONFIRMATION = "ASK_CONFIRMATION"
    FINISH = "FINISH"


ACTIONS: tuple[str, ...] = (Action.AUTHENTICATE.value, Action.ASK_CONFIRMATION.value, *TOOL_NAMES, Action.FINISH.value)


def missing_required_params(tool_name: str, params: dict[str, Any]) -> list[str]:
    return [param for param in TOOLS[tool_name].required_params if param not in params]

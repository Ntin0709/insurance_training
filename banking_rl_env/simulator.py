"""Deterministic banking tool simulator for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4
import re

from banking_rl_env.tools import TOOLS, missing_required_params


def error(error_code: str, error_message: str, retriable: bool = False, retry_after_seconds: int = 0) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "error_message": error_message,
        "retriable": retriable,
        "retry_after_seconds": retry_after_seconds,
    }


@dataclass
class BankingToolSimulator:
    """Executes tool calls with auth, confirmation, prerequisite, and idempotency checks."""

    authenticated: bool = False
    confirmed: bool = False
    called_tools: set[str] = field(default_factory=set)
    idempotency_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def authenticate(self) -> dict[str, Any]:
        self.authenticated = True
        return {"authenticated": True, "session_expires_at": (datetime.utcnow() + timedelta(minutes=15)).isoformat()}

    def ask_confirmation(self) -> dict[str, Any]:
        self.confirmed = True
        return {"confirmed": True, "confirmed_at": datetime.utcnow().isoformat()}

    def call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOLS:
            return error("VALIDATION_ERROR", f"Unknown tool: {tool_name}")

        spec = TOOLS[tool_name]
        missing = missing_required_params(tool_name, params)
        if missing:
            return error("VALIDATION_ERROR", f"Missing required parameter(s): {', '.join(missing)}")

        validation_error = self._validate(tool_name, params)
        if validation_error:
            return validation_error

        if spec.requires_auth and not self.authenticated:
            return error("AUTH_FAILED", "Authentication is required before calling this tool")

        if spec.requires_confirmation and not self.confirmed:
            return error("VALIDATION_ERROR", "User confirmation is required before calling this tool")

        missing_prereqs = [name for name in spec.prerequisites if name not in self.called_tools]
        if missing_prereqs:
            return error("VALIDATION_ERROR", f"Missing prerequisite tool(s): {', '.join(missing_prereqs)}")

        if spec.write_tool:
            key = str(params["idempotency_key"])
            if key in self.idempotency_cache:
                original = dict(self.idempotency_cache[key])
                original["duplicate_request"] = True
                return original

        result = self._execute(tool_name, params)
        self.called_tools.add(tool_name)
        self.outputs[tool_name] = result

        if spec.write_tool:
            self.idempotency_cache[str(params["idempotency_key"])] = result

        return result

    def _validate(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
        enum_values = {
            "account_type": {"SAVINGS", "CURRENT", "OD"},
            "status": {"ACTIVE", "CLOSED", "MATURED"},
            "transaction_type": {"CREDIT", "DEBIT", "ALL"},
            "document_type": {"PAN", "AADHAAR", "PASSPORT", "VOTERID"},
            "purity": {"18K", "22K", "24K"},
            "payout_type": {"CUMULATIVE", "MONTHLY", "QUARTERLY"},
            "compounding_frequency": {"QUARTERLY", "MONTHLY", "ANNUAL"},
            "reason": {"EMERGENCY", "REINVESTMENT", "OTHER"},
            "product_category": {"FD", "RD", "LOAN", "GOLD", "ACCOUNT"},
            "circular_type": {"MASTER", "NOTIFICATION", "GUIDELINE"},
            "customer_type": {"GENERAL", "SENIOR_CITIZEN", "NRI"},
            "product_type": {"HOME", "PERSONAL", "GOLD", "LAP", "AUTO", "EDUCATION"},
            "customer_segment": {"SALARIED", "SELF_EMPLOYED", "BUSINESS"},
            "request_type": {"CHEQUE_BOOK", "STATEMENT", "NOMINEE_UPDATE", "ADDRESS_CHANGE"},
            "priority": {"NORMAL", "URGENT"},
            "format": {"PDF", "EMAIL", "BOTH"},
            "complaint_category": {"SERVICE", "CHARGE", "FRAUD", "STAFF", "DIGITAL"},
            "severity": {"LOW", "MEDIUM", "HIGH"},
            "preferred_contact": {"EMAIL", "SMS", "CALL"},
        }
        for param, values in enum_values.items():
            if param in params and params[param] not in values:
                return error("VALIDATION_ERROR", f"{param} must be one of: {', '.join(sorted(values))}")

        if tool_name in {"calc_fd_maturity", "book_new_fd"}:
            principal = float(params.get("principal", params.get("principal_amount", 0)))
            if principal <= 0:
                return error("VALIDATION_ERROR", "principal amount must be positive")
            if principal > 250000 and tool_name == "book_new_fd":
                return error("INSUFFICIENT_BALANCE", "Available balance is insufficient for FD booking")

        if tool_name == "get_transaction_history":
            from_date = date.fromisoformat(str(params["from_date"]))
            to_date = date.fromisoformat(str(params["to_date"]))
            if from_date > to_date:
                return error("VALIDATION_ERROR", "from_date cannot be after to_date")
            if (to_date - from_date).days > 1095:
                return error("VALIDATION_ERROR", "Maximum transaction lookback is 3 years")

        if TOOLS[tool_name].write_tool:
            key = str(params.get("idempotency_key", ""))
            uuid4ish = re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", key)
            if not uuid4ish:
                return error("VALIDATION_ERROR", "idempotency_key must be a UUID v4")

        return None

    def _execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        today = date(2026, 6, 4)

        if tool_name == "generate_uuid":
            count = min(int(params.get("count", 1)), 10)
            return {"uuids": [str(uuid4()) for _ in range(count)], "version": "v4", "generated_at": datetime.utcnow().isoformat()}

        if tool_name == "get_account_balance":
            return {
                "customer_id": params["customer_id"],
                "account_type": params.get("account_type", "SAVINGS"),
                "available_balance": 250000.0,
                "hold_amount": 0.0,
                "currency": "INR",
                "last_updated": datetime.utcnow().isoformat(),
            }

        if tool_name == "get_fd_details":
            return {
                "fd_number": params.get("fd_number", "FD001"),
                "principal_amount": 100000.0,
                "interest_rate": 6.75,
                "tenure": {"value": 365, "unit": "DAYS"},
                "start_date": "2025-07-01",
                "maturity_date": "2026-07-01",
                "maturity_amount": 106923.31,
                "payout_type": "CUMULATIVE",
                "status": params.get("status", "ACTIVE"),
                "nominee": "Registered nominee",
            }

        if tool_name == "get_rd_details":
            return {
                "rd_number": params.get("rd_number", "RD001"),
                "monthly_installment": 5000.0,
                "interest_rate": 6.5,
                "tenure_months": 24,
                "start_date": "2025-01-01",
                "maturity_date": "2027-01-01",
                "total_deposited": 85000.0,
                "maturity_amount": 126770.83,
                "installments_paid": 17,
                "installments_pending": 7,
                "status": params.get("status", "ACTIVE"),
            }

        if tool_name == "get_loan_account_details":
            return {
                "customer_id": params["customer_id"],
                "loan_id": params.get("loan_id", "LN001"),
                "loan_type": "HOME",
                "product_name": "Home Loan Advantage",
                "sanctioned_amount": 5000000.0,
                "disbursed_amount": 5000000.0,
                "outstanding_principal": 4210000.0,
                "interest_rate": 8.65,
                "rate_type": "FLOATING",
                "tenure_months": 240,
                "emi_amount": 43860.0,
                "disbursement_date": "2023-04-10",
                "maturity_date": "2043-04-10",
                "overdue_amount": 0.0,
                "overdue_days": 0,
                "next_emi_date": "2026-07-05",
                "next_emi_amount": 43860.0,
                "loan_status": "ACTIVE",
                "co_applicants": [] if not params.get("include_co_applicants") else [{"name": "Co-applicant"}],
                "collateral": [] if not params.get("include_collateral") else [{"type": "PROPERTY"}],
            }

        if tool_name == "get_transaction_history":
            transactions = [
                {
                    "txn_id": "TXN001",
                    "date": "2026-05-22",
                    "description": "UPI debit",
                    "amount": 2499.0,
                    "type": "DEBIT",
                    "balance_after": 247501.0,
                    "channel": "UPI",
                    "reference_no": "BANKREF001",
                    "utr_number": "UTR001",
                    "upi_ref_id": "UPI001",
                },
                {
                    "txn_id": "TXN002",
                    "date": "2026-05-24",
                    "description": "Salary credit",
                    "amount": 100000.0,
                    "type": "CREDIT",
                    "balance_after": 347501.0,
                    "channel": "NEFT",
                    "reference_no": "BANKREF002",
                    "utr_number": "UTR002",
                    "upi_ref_id": "",
                },
            ]
            txn_type = params.get("transaction_type", "ALL")
            if txn_type != "ALL":
                transactions = [txn for txn in transactions if txn["type"] == txn_type]
            limit = int(params.get("limit", 50))
            return {"account_id": params["account_id"], "total_records": len(transactions), "transactions": transactions[:limit]}

        if tool_name == "get_kyc_status":
            return {
                "customer_id": params["customer_id"],
                "kyc_status": "COMPLETE",
                "kyc_expiry_date": "2028-03-31",
                "verified_documents": ["PAN", "AADHAAR"],
                "pending_documents": [],
                "last_updated": datetime.utcnow().isoformat(),
                "re_kyc_required": False,
            }

        if tool_name == "get_foreclosure_quote":
            return {
                "loan_id": params.get("loan_id", "LN001"),
                "foreclosure_date": params.get("foreclosure_date", str(today)),
                "outstanding_principal": 4210000.0,
                "accrued_interest": 29900.0,
                "foreclosure_charges": 0.0,
                "gst_on_charges": 0.0,
                "total_payable": 4239900.0,
                "quote_valid_till": str(today + timedelta(days=7)),
            }

        if tool_name == "calc_fd_maturity":
            principal = float(params["principal"])
            rate = float(params["interest_rate"]) + (0.5 if params.get("senior_citizen") else 0.0)
            tenure = int(params["tenure"])
            maturity = principal * ((1 + rate / 400) ** (tenure / 91.25))
            return {
                "principal": principal,
                "interest_rate": float(params["interest_rate"]),
                "effective_rate": round(rate, 2),
                "tenure_days": tenure,
                "maturity_amount": round(maturity, 2),
                "total_interest": round(maturity - principal, 2),
                "tds_applicable": maturity - principal > 40000,
                "payout_schedule": [],
            }

        if tool_name == "calc_premature_penalty":
            principal = 100000.0
            interest_earned = 3200.0
            penalty_amount = 500.0
            return {
                "fd_number": params["fd_number"],
                "original_rate": 6.75,
                "applicable_rate": 5.75,
                "penalty_rate": 1.0,
                "principal": principal,
                "interest_earned": interest_earned,
                "penalty_amount": penalty_amount,
                "net_payout": principal + interest_earned - penalty_amount,
            }

        if tool_name == "calc_rd_maturity":
            installment = float(params["monthly_installment"])
            months = int(params["tenure_months"])
            deposited = installment * months
            interest = deposited * float(params["interest_rate"]) / 100 * (months + 1) / 24
            return {
                "monthly_installment": installment,
                "total_deposited": round(deposited, 2),
                "total_interest": round(interest, 2),
                "maturity_amount": round(deposited + interest, 2),
                "effective_yield": float(params["interest_rate"]),
            }

        if tool_name == "calc_emi":
            principal = float(params["principal"])
            monthly_rate = float(params["interest_rate"]) / 1200
            months = int(params["tenure_months"])
            emi = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
            fee = float(params.get("processing_fee", 0))
            return {
                "emi_amount": round(emi, 2),
                "total_interest": round(emi * months - principal, 2),
                "total_payable": round(emi * months + fee, 2),
                "effective_irr": float(params["interest_rate"]),
                "processing_fee": fee,
                "first_emi_date": params.get("start_date", str(today + timedelta(days=30))),
            }

        if tool_name == "calc_gold_ltv":
            rates = {"18K": 7200.0, "22K": 8800.0, "24K": 9600.0}
            rate = rates[str(params["purity"])]
            gross = float(params["gold_weight_grams"]) * rate
            ltv = min(float(params.get("ltv_percentage", 75.0)), 75.0)
            return {
                "gold_weight_grams": float(params["gold_weight_grams"]),
                "purity": params["purity"],
                "gold_rate_per_gram": rate,
                "gross_value": round(gross, 2),
                "ltv_percentage": ltv,
                "eligible_loan_amount": round(gross * ltv / 100, 2),
                "rbi_cap_applied": float(params.get("ltv_percentage", 75.0)) > 75.0,
            }

        if tool_name == "get_branch_info":
            return {
                "total_results": 1,
                "branches": [
                    {
                        "branch_code": "MUM001",
                        "branch_name": "Mumbai Main",
                        "address": "Fort, Mumbai",
                        "city": "Mumbai",
                        "pincode": "400001",
                        "ifsc": "BANK000001",
                        "phone": "022-40000000",
                        "timings": "10:00-16:00",
                        "services": ["LOCKER", "ATM", "FOREX"],
                        "is_atm_available": True,
                    }
                ],
            }

        if tool_name == "get_gold_rate_today":
            rates = {"18K": 7200.0, "22K": 8800.0, "24K": 9600.0}
            purity = params.get("purity")
            selected = [purity] if purity else ["18K", "22K", "24K"]
            return {
                "date": str(today),
                "currency": "INR",
                "valid_till": datetime.combine(today, datetime.max.time()).isoformat(),
                "source": "simulated_market_feed",
                "rates": [
                    {"purity": item, "rate_per_gram": rates[item], "rate_per_10g": rates[item] * 10}
                    for item in selected
                ],
            }

        if tool_name == "search_product_kb":
            return {
                "query": params["query"],
                "results": [
                    {
                        "title": "Fixed Deposit Features",
                        "snippet": "FDs support cumulative, monthly, and quarterly payout options.",
                        "source_doc": "product_kb_fd",
                        "relevance_score": 0.91,
                        "last_updated": "2026-06-01",
                    }
                ],
            }

        if tool_name == "search_rbi_circular":
            return {
                "total_results": 1,
                "results": [
                    {
                        "circular_id": "RBI-2025-GL-001",
                        "title": "Gold loan loan-to-value guidance",
                        "issued_date": "2025-04-01",
                        "category": "Gold Loans",
                        "summary": "LTV caps apply to loans secured by gold ornaments.",
                        "url": "https://rbi.example.invalid/circular/RBI-2025-GL-001",
                    }
                ],
            }

        if tool_name == "get_fd_rate_card":
            return {
                "effective_date": params.get("effective_date", str(today)),
                "slabs": [
                    {
                        "tenure_label": "1 year",
                        "min_days": 365,
                        "max_days": 389,
                        "general_rate": 6.75,
                        "senior_citizen_rate": 7.25,
                        "nri_rate": 6.75,
                    }
                ],
            }

        if tool_name == "get_loan_product_details":
            return {
                "product_type": params["product_type"],
                "interest_rate_range": {"min": 8.5, "max": 12.0},
                "min_loan_amount": 100000.0,
                "max_loan_amount": 10000000.0,
                "tenure_range": {"min_months": 12, "max_months": 300},
                "processing_fee": "Up to 1% plus taxes",
                "prepayment_charges": "As per rate type and product",
                "eligibility_criteria": ["Valid KYC", "Income proof", "Credit assessment"],
                "documents_required": ["PAN", "Address proof", "Bank statement"],
            }

        if tool_name == "book_new_fd":
            maturity = self._execute(
                "calc_fd_maturity",
                {
                    "principal": params["principal_amount"],
                    "interest_rate": 6.75,
                    "tenure": params["tenure"],
                    "payout_type": params.get("payout_type", "CUMULATIVE"),
                },
            )
            return {
                "fd_number": "FDNEW001",
                "principal_amount": float(params["principal_amount"]),
                "interest_rate": 6.75,
                "maturity_date": str(today + timedelta(days=int(params["tenure"]))),
                "maturity_amount": maturity["maturity_amount"],
                "debit_account": params["source_account_id"],
                "booking_timestamp": datetime.utcnow().isoformat(),
                "receipt_ref": "FDRCPT001",
            }

        if tool_name == "request_premature_closure":
            return {
                "request_id": "FDCLS001",
                "fd_number": params["fd_number"],
                "net_payout": 102700.0,
                "tds_deducted": 0.0,
                "penalty_applied": 500.0,
                "credit_account": params["credit_account_id"],
                "expected_credit_date": str(today + timedelta(days=1)),
                "status": "SUBMITTED",
            }

        if tool_name == "log_complaint":
            return {
                "complaint_id": "CMP001",
                "category": params["complaint_category"],
                "severity": params.get("severity", "HIGH" if params["complaint_category"] == "FRAUD" else "MEDIUM"),
                "status": "LOGGED",
                "tat_days": 3,
                "resolution_deadline": str(today + timedelta(days=3)),
                "escalation_path": "Branch Manager > Nodal Officer",
                "created_at": datetime.utcnow().isoformat(),
            }

        if tool_name == "raise_service_request":
            return {
                "sr_number": "SR001",
                "request_type": params["request_type"],
                "status": "RAISED",
                "estimated_resolution": str(today + timedelta(days=5)),
                "created_at": datetime.utcnow().isoformat(),
                "channel": "ASSISTANT",
            }

        if tool_name == "generate_interest_certificate":
            return {
                "certificate_ref": "CERT001",
                "financial_year": params["financial_year"],
                "total_interest": 28750.0,
                "tds_deducted": 0.0,
                "accounts_covered": params.get("account_ids", ["SA001", "FD001"]),
                "generated_at": datetime.utcnow().isoformat(),
                "download_url": "https://bank.example.invalid/certificates/CERT001",
                "expiry": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            }

        return {"tool": tool_name, "status": "OK", "generated_at": datetime.utcnow().isoformat()}

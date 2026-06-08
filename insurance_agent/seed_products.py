"""Seed products used when scraping has not been run yet."""

from __future__ import annotations

from insurance_agent.models import InsuranceProduct


SEED_PRODUCTS: tuple[InsuranceProduct, ...] = (
    InsuranceProduct(
        product_id="sbi_health_alpha",
        name="SBI General Health Alpha",
        category="health",
        provider="SBI General Insurance",
        suitable_for=("individual", "family_floater", "high_sum_insured", "maternity_optional"),
        key_features=(
            "Hospitalization cover with multiple sum insured options",
            "Optional organ donor and maternity-related covers may be available depending on plan choices",
            "Network hospital and claim process should be checked before purchase",
        ),
        exclusions_or_waiting=(
            "Waiting periods and exclusions apply",
            "Pre-policy medical check-up may apply by age and sum insured",
        ),
        sum_insured_options=("5L", "7.5L", "10L", "15L", "20L", "25L", "50L", "1Cr+"),
        source_ids=("sbi_health_alpha_prospectus",),
    ),
    InsuranceProduct(
        product_id="sbi_motor_private_car",
        name="SBI General Motor Private Car Insurance",
        category="motor",
        provider="SBI General Insurance",
        suitable_for=("car_owner", "third_party_liability", "own_damage"),
        key_features=(
            "Covers vehicle loss or damage depending on selected plan",
            "Third-party liability cover is relevant for vehicle owners",
            "Add-ons and claim garage network should be checked",
        ),
        source_ids=("sbi_motor_private_car",),
    ),
    InsuranceProduct(
        product_id="sbi_travel",
        name="SBI General Travel Insurance",
        category="travel",
        provider="SBI General Insurance",
        suitable_for=("international_travel", "domestic_travel", "frequent_traveller"),
        key_features=(
            "Travel cover for trip-related medical and non-medical risks depending on plan",
            "Destination, trip duration, age, and pre-existing illness disclosures matter",
        ),
        exclusions_or_waiting=("Adventure sports and pre-existing conditions may have restrictions",),
        source_ids=("sbi_travel",),
    ),
    InsuranceProduct(
        product_id="sbi_personal_accident",
        name="SBI General Personal Accident Insurance",
        category="personal_accident",
        provider="SBI General Insurance",
        suitable_for=("income_protection", "accidental_death", "disability_cover"),
        key_features=(
            "Accident-linked death and disability protection depending on selected cover",
            "Useful as a supplement, not a replacement for health insurance",
        ),
        source_ids=("sbi_general_products",),
    ),
    InsuranceProduct(
        product_id="sbi_home_insurance",
        name="SBI General Home Insurance",
        category="home",
        provider="SBI General Insurance",
        suitable_for=("home_owner", "contents_cover", "property_protection"),
        key_features=(
            "Protects house structure and/or contents depending on selected plan",
            "Relevant for homeowners with property risk exposure",
        ),
        source_ids=("sbi_cyber_insurance",),
    ),
    InsuranceProduct(
        product_id="sbi_cyber_insurance",
        name="SBI General Cyber Insurance",
        category="cyber",
        provider="SBI General Insurance",
        suitable_for=("digital_payment_user", "online_banking_user", "upi_user"),
        key_features=(
            "Cyber-risk protection for eligible digital incidents depending on policy terms",
            "Useful for customers with heavy online banking and card usage",
        ),
        source_ids=("sbi_general_home",),
    ),
)

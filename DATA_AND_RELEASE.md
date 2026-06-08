# Data and Release Bundle

Raw generated JSONL datasets are not committed because several files exceed GitHub's 100 MB file limit.

The runnable training/eval bundle is committed here:

```text
release/gemma3_insurance_training_eval_bundle.zip
```

That zip contains the training/eval code, environment setup, validated training splits needed for the baseline run, and the smoke-test launcher.

Local full data remains on the generation machine under:

```text
/home/npci-admin/Downloads/RL_ENV/data/insurance/
```

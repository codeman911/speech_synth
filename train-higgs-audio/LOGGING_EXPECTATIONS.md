# Strategic Logging Expectations

## When to Expect Logs

With the updated strategic logging implementation, you should see logs at the following intervals:

1. **Step 1**: Initial debug log to confirm callbacks are working
2. **Every N steps**: Where N is the value of `--strategic_logging_steps` (default: 100)

For your current command with `--strategic_logging_steps 100`, you should see logs at:
- Step 1 (debug)
- Step 100
- Step 200
- Step 300
- And so on...

## Log Format

The logs will appear with a timestamp prefix:

```
[2023-08-15 14:32:45] STRATEGIC LOG:
=== Zero-Shot Voice Cloning Training Log - Step 100 ===
...
```

## Troubleshooting

If you don't see logs:

1. **Check the step count**: Make sure enough steps have passed
2. **Check for errors**: Look for "STRATEGIC LOG ERROR" messages
3. **Check for debug messages**: Look for "STRATEGIC LOG DEBUG" messages at step 1

## Your Current Training Status

Based on your output:
```
{'loss': 7.2558, 'grad_norm': 0.30584660172462463, 'learning_rate': 5e-06, 'epoch': 0.0}
  0%|▏| 117/143361 [01:11<19:16:40,  2.06it/s]
```

You're currently at step 117. You should see the first strategic log when you reach step 200 (since step 100 has already passed).

The logs will show:
- Input analysis (tensor shapes, decoded text)
- Output analysis (loss, predictions vs targets)
- Shared attention verification
- Zero-shot capability metrics
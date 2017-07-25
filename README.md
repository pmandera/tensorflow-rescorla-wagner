# Rescorla-Wagner model

This implements the
[Rescorla-Wagner](http://www.scholarpedia.org/article/Rescorla-Wagner_model)
model in TensorFlow.

## Example usage

To fetch datasets included in the
[NDL](https://cran.r-project.org/web/packages/ndl/index.html) run:

```bash
./get-ndl-data.sh
```

Next, the training can be started:

```bash
./rw.py --data_file datasets/ndl/serbian.txt \
  --alpha_beta 0.001 --n_steps 1e7 --add_bias
```

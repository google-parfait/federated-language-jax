# DUCC Patches

1.  Clone the repository.

    ```shell
    $ git clone https://github.com/openxla/xla.git "/tmp/xla"
    $ cd "/tmp/xla"
    ```

1.  Checkout the commit.

    ```shell
    $ git checkout <COMMIT>
    ```

1.  Create `bazel_deps.patch`.

    Make the changes.

    ```shell
    $ find "third_party/ducc" -type f -print0 | xargs -0 \
        sed --in-place \
        -e 's/eigen_archive\/\/:eigen3/@eigen\/\/:eigen/g'
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/ducc/bazel_deps.patch"
    ```

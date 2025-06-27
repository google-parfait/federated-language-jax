# TSL Patches

IMPORTANT: The TSL repository is being migrated from TensorFlow and is currently
provided by the XLA repository
https://github.com/openxla/xla/tree/main/third_party/tsl.

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
    $ buildozer 'replace deps @eigen_archive//:eigen3 @eigen//:eigen' //third_party/tsl/...:*
    $ buildozer 'substitute deps @com_google_absl//(.*) @abseil-cpp//${1}' //third_party/tsl/...:*
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/tsl/bazel_deps.patch"
    ```

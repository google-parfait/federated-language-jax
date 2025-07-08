# XLA Patches

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
    $ buildozer 'replace deps @eigen_archive//:eigen3 @eigen//:eigen' //xla/...:*
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/xla/bazel_deps.patch"
    ```

1.  Create `zlib.patch`.

    Make the changes.

    ```shell
    $ sed --in-place \
        -e 's/hdrs = \[\"zlib\.h\"\],/hdrs = ["zconf.h", "zlib.h"],/g' \
        "third_party/zlib.BUILD"
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/xla/zlib.patch"
    ```

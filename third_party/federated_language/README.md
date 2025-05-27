# Federated Language Patches

1.  Clone the repository.

    ```shell
    $ git clone https://github.com/google-parfait/federated-language.git "/tmp/federated-language"
    $ cd "/tmp/federated-language"
    ```

1.  Checkout the commit.

    ```shell
    $ git checkout <COMMIT>
    ```

1.  Create `proto_library_loads.patch`.

    Make the changes.

    ```shell
    $ buildozer 'replace_load @rules_cc//cc:defs.bzl cc_proto_library' //...:*
    $ buildozer 'replace_load @rules_python//python:proto.bzl py_proto_library' //...:*
    $ buildozer 'fix unusedLoads' //...:*
    $ find "." -type f -print0 | xargs -0 \
        sed --in-place \
        -e '/@protobuf\/\/bazel:proto_library.bzl/d'
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/federated_language/proto_library_loads.patch"
    ```

1.  Create `structure_visibility.patch`.

    Make the changes.

    ```shell
    $ buildozer 'add visibility //visibility:public' //federated_language/common_libs:structure
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/federated_language/structure_visibility.patch"
    ```

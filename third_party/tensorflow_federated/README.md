# TensorFlow Federated Patches

1.  Clone the repository.

    ```shell
    $ git clone https://github.com/google-parfait/tensorflow-federated.git "/tmp/tensorflow-federated"
    $ cd "/tmp/tensorflow-federated"
    ```

1.  Checkout the commit.

    ```shell
    $ git checkout <COMMIT>
    ```

1.  Create `bazel_deps.patch`.

    Make the changes.

    ```shell
    $ buildozer 'replace_load @rules_python//python:proto.bzl py_proto_library' //...:*
    $ buildozer 'fix unusedLoads' //...:*
    $ buildozer 'substitute deps @org_tensorflow//tensorflow/compiler/xla([/:].*) @xla//xla${1}' //...:*
    $ buildozer 'substitute deps @org_tensorflow//tensorflow/tsl/(.*) @tsl//tsl/${1}' //...:*
    $ buildozer \
        'add deps @federated_language_jax_pypi//grpcio' \
        //tensorflow_federated/python/core/impl/executors:executors_errors
    $ find "." -type f -print0 | xargs -0 \
        sed --regexp-extended --in-place \
        -e 's/#include "third_party\/eigen3\/Eigen\//#include "Eigen\//g' \
        -e 's/#include "include\/grpcpp\/impl\/channel_interface.h"/#include "grpcpp\/impl\/codegen\/channel_interface.h"/g' \
        -e 's/#include "include\/grpcpp\//#include "grpcpp\//g' \
        -e 's/#include "tensorflow\/tsl\//#include "tsl\//g' \
        -e 's/#include "tensorflow\/compiler\/xla\//#include "xla\//g'
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/third_party/py/federated_language_jax/third_party/tensorflow_federated/bazel_deps.patch"
    ```


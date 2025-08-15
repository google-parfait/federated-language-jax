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

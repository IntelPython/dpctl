.. rubric:: Use :meth:`usm_ndarray.to_device` to migrate array to different device

.. code-block:: python
    :caption: Migrate array to a different device

        from dpctl import tensor

        a = tensor.zeros(100, device="cpu")
        b = a.to_device("gpu")


.. rubric:: Use :meth:`usm_ndarray.device` to specify placement of new array

.. code-block:: python
    :caption: Create an USM-device empty array on the same device as another array

        from dpctl import tensor

        d = tensor.eye(100)
        u = tensor.full(d.shape, fill_value=0.5, usm_type="device", device=d.device)

.. rubric:: Use :meth:`usm_ndarray.mT` to transpose matrices in a array thought of as a stack of matrices

.. code-block:: python
    :caption: Transpose an array

        from dpctl import tensor

        # create stack of matrices
        proto = tensor.asarray([[2, 1], [3, 4]])
        ar = tensor.tile(proto, (5, 10, 10))

        # transpose each matrix in the stack
        arT = ar.mT

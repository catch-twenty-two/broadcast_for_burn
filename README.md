
Broadcast two tensors with potentially different static ranks to a common rank in the *Burn* Framework

See the PyTorch website for a good description:
https://docs.pytorch.org/docs/stable/notes/broadcasting.html

## General semantics

  Two tensors are “broadcastable” if the following rules hold:

- Each tensor has at least one dimension.
    
- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

 # Rust/Burn Framework Syntax
 ```ignore
 broadcast!(
     a: Tensor<Backend, RANK_A>,
     b: Tensor<RANK_B>
 )
 ```

 # Parameters
 - `a`: Identifier for the first tensor variable.
 - `Backend`: The backend to use
 - `RANK_A`: The static rank of the first tensor (e.g., `2`, `3`, etc.).

 - `b`: Identifier for the second tensor variable.
 - `RANK_B`: The static rank of the second tensor.

 # Example
 ```rust
     let device = &NdArrayDevice::default();
     type B = NdArray<f32>;

     let a = Tensor::<B, 3>::from_data(
         [
             [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
             [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
         ],
         device,
     );

     let b = Tensor::<B, 2>::from_data([[4, 11, 10, 5]], device);

     let (a, b) = broadcast!(a:Tensor<B, 3>, b:Tensor<2>);

     let a_add_b = a.add(b);
 
 // Output:
 // Tensor {
 //   data:
 // [[[ 6.0, 19.0, 17.0,  7.0],
 //   [13.0, 25.0, 23.0, 17.0],
 //   [13.0, 25.0, 23.0, 17.0]],
 //  [[ 6.0, 19.0, 17.0,  7.0],
 //   [13.0, 25.0, 23.0, 17.0],
 //   [13.0, 25.0, 23.0, 17.0]]],
 //   shape:  [2, 3, 4],
 //   device:  Cpu,
 //   backend:  "ndarray",
 //   kind:  "Float",
 //   dtype:  "f32",
 // }
 ```

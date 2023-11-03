import torch


def zeros_tensor(shape):
    """
    Returns a 2-dimensional tensor filled with all zeros.

    Args:
    shape (tuple): The shape of the desired tensor, e.g., (rows, columns).

    Returns:
    torch.Tensor: A tensor with all zeros.
    """
    return torch.zeros(shape)


def ones_tensor(shape):
    """
    Returns a 2-dimensional tensor filled with all ones.

    Args:
    shape (tuple): The shape of the desired tensor, e.g., (rows, columns).

    Returns:
    torch.Tensor: A tensor with all ones.
    """
    return torch.ones(shape)


def random_tensor(shape):
    """
    Returns a 2-dimensional tensor filled with random values.

    Args:
    shape (tuple): The shape of the desired tensor, e.g., (rows, columns).

    Returns:
    torch.Tensor: A tensor with random values.
    """
    return torch.rand(shape)


def add_tensors(tensor1, tensor2):
    """
    Returns the sum of two 2-dimensional tensors.

    Args:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.

    Returns:
    torch.Tensor: The sum of the two input tensors.
    """
    return tensor1 + tensor2


def multiply_tensors(tensor1, tensor2):
    """
    Returns the element-wise multiplication of two 2-dimensional tensors.

    Args:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.

    Returns:
    torch.Tensor: The element-wise multiplication of the two input tensors.
    """
    return tensor1 * tensor2


def main():
    tensor_shape = (2, 3)
    zeros = zeros_tensor(tensor_shape)
    ones = ones_tensor(tensor_shape)
    random_values = random_tensor(tensor_shape)

    tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    tensor_b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    sum_result = add_tensors(tensor_a, tensor_b)
    multiply_result = multiply_tensors(tensor_a, tensor_b)

    print("Zeros Tensor:\n", zeros)
    print("Ones Tensor:\n", ones)
    print("Random Tensor:\n", random_values)
    print("Sum of Tensors:\n", sum_result)
    print("Element-wise Multiplication of Tensors:\n", multiply_result)


if __name__ == "__main__":
    main()

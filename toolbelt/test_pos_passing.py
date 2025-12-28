import torch

# Simular lo que pasa
pos_ids_batch = torch.LongTensor([[0, 1, 2, 3, 4, 127, 127, ...],
                                   [0, 1, 2, 3, 127, 127, ...]])

print("pos_ids_batch shape:", pos_ids_batch.shape)
print("pos_ids_batch[0]:", pos_ids_batch[0])
print("Todos iguales?", (pos_ids_batch[0] == pos_ids_batch[1]).all())

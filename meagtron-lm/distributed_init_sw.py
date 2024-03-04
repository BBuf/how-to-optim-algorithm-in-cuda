world_size: int = 16
tensor_model_parallel_size = 2
context_parallel_size = 2
pipeline_model_parallel_size = 4
expert_model_parallel_size = 1
moe_expert_tensor_parallelism = False
moe_expert_data_parallelism = False

data_parallel_size: int = world_size // (
    tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
)
tensor_or_replicate_parallel_size = tensor_model_parallel_size
num_tensor_or_replicate_parallel_groups: int = world_size // tensor_or_replicate_parallel_size
tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
num_expert_groups: int = data_parallel_size // expert_model_parallel_size

tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp


if (
    world_size
    % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
    != 0
):
    raise RuntimeError(
        f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
        f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
        f"x context_parallel_size ({context_parallel_size})"
    )


if data_parallel_size % expert_model_parallel_size != 0:
    raise RuntimeError(
        f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
    )

if expert_model_parallel_size > 1 and context_parallel_size > 1:
    raise RuntimeError(
        f"combination of expert model prallellism and context parallelism is not supported"
    )

num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

all_data_parallel_group_ranks_with_cp = []
all_data_parallel_group_ranks = []
for i in range(pipeline_model_parallel_size):
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i + 1) * num_pipeline_model_parallel_groups
    for j in range(context_parallel_size * tensor_model_parallel_size):
        ranks = range(
            start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
        )
        all_data_parallel_group_ranks.append(list(ranks))
    for j in range(tensor_model_parallel_size):
        ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
        all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))

print('data_parallel groups: ')
print(all_data_parallel_group_ranks)
print('='*50)

expert_parallel_group_ranks = []

assert not (moe_expert_tensor_parallelism and moe_expert_data_parallelism), \
    'cannot use both expert tensor parallelism and expert data parallelism'
if moe_expert_tensor_parallelism:
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_or_replicate_parallel_size):
            ranks = range(start_rank + j, end_rank,
                            tensor_or_replicate_parallel_size)
            expert_parallel_group_ranks.append(list(ranks))
elif moe_expert_data_parallelism:
    for i in range(num_tensor_or_replicate_parallel_groups):
        ranks = range(i * tensor_or_replicate_parallel_size, (i + 1) * tensor_or_replicate_parallel_size)
        expert_parallel_group_ranks.append(list(ranks))
else:
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        ranks = range(start_rank, end_rank)
        expert_parallel_group_ranks.append(list(ranks))

print('expert_parallel_group_ranks: ')
print(expert_parallel_group_ranks)
print('='*50)

context_parallel_group_ranks = []

for i in range(pipeline_model_parallel_size):
    for j in range(data_parallel_size):
        start_rank = (
            i * num_pipeline_model_parallel_groups
            + j * tensor_model_parallel_size * context_parallel_size
        )
        end_rank = (
            i * num_pipeline_model_parallel_groups
            + (j + 1) * tensor_model_parallel_size * context_parallel_size
        )
        for k in range(tensor_model_parallel_size):
            ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
            context_parallel_group_ranks.append(list(ranks))

print('context_parallel_group_ranks: ')
print(context_parallel_group_ranks)
print('='*50)

tensor_model_parallel_group_ranks = []
for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
    tensor_model_parallel_group_ranks.append(list(ranks))
    
print('tensor_model_parallel_group_ranks: ')
print(tensor_model_parallel_group_ranks)
print('='*50)

pipline_parallel_group_ranks = []

for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size, num_pipeline_model_parallel_groups)
    pipline_parallel_group_ranks.append(list(ranks))

print('pipline_parallel_group_ranks: ')
print(pipline_parallel_group_ranks)
print('='*50)


model_parallel_groups = []
# Build the model-parallel groups.

for i in range(data_parallel_size * context_parallel_size):
    ranks = [
        data_parallel_group_ranks_with_cp[i]
        for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
    ]
    model_parallel_groups.append(ranks)

print('model_parallel_groups: ')
print(model_parallel_groups)
print('='*50)

# Build the tensor + data parallel groups.
tensor_and_data_paralle_group_with_cp_groups = []
tensor_and_data_parallel_groups = []

for i in range(num_tensor_and_data_groups_with_cp):
    start_rank = i * tensor_and_data_group_size_with_cp
    end_rank = start_rank + tensor_and_data_group_size_with_cp
    ranks = range(start_rank, end_rank)
    tensor_and_data_paralle_group_with_cp_groups.append(ranks)

    for j in range(context_parallel_size):
        ranks = []
        for k in range(data_parallel_size):
            start_rank = (
                i * tensor_and_data_group_size_with_cp
                + j * tensor_model_parallel_size
                + k * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = start_rank + tensor_model_parallel_size
            ranks = ranks + list(range(start_rank, end_rank))
        tensor_and_data_parallel_groups.append(ranks)

print('tensor_and_data_paralle_group_with_cp_groups: ')
print(tensor_and_data_paralle_group_with_cp_groups)
print('='*50)
print('tensor_and_data_parallel_groups: ')
print(tensor_and_data_parallel_groups)
print('='*50)

# Build the tensor + expert parallel groups

tensor_and_expert_parallel_groups = []
data_modulo_expert_parallel_group = []
for i in range(num_tensor_and_data_groups):
    for j in range(num_expert_groups):
        start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
        end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
        ranks = range(start_rank, end_rank)
        tensor_and_expert_parallel_groups.append(list(ranks))

for i in range(num_tensor_and_data_groups):
    start_rank = i * tensor_and_data_group_size
    end_rank = (i + 1) * tensor_and_data_group_size
    for j in range(tensor_and_expert_group_size):
        ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
        data_modulo_expert_parallel_group.append(list(ranks))

print('tensor_and_expert_parallel_groups: ')
print(tensor_and_expert_parallel_groups)
print('='*50)
print('data_modulo_expert_parallel_group: ')
print(data_modulo_expert_parallel_group)
print('='*50)

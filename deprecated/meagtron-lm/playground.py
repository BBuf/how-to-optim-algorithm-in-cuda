import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from descriptions import basic_texts, descriptions

def generate_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    """
    Generate data parallel groups based on the provided parallelism parameters.
    """
    assert world_size % (pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size) == 0, "world_size must be divisible by the product of pipeline_model_parallel_size, tensor_model_parallel_size, and context_parallel_size"
    data_parallel_group_ranks = []
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(
                start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
            )
            data_parallel_group_ranks.append(list(ranks))
    return data_parallel_group_ranks

def generate_context_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    """
    Generate data parallel groups considering context parallelism.
    """
    assert world_size % (pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size) == 0, "world_size must be divisible by the product of pipeline_model_parallel_size, tensor_model_parallel_size, and context_parallel_size"
    all_data_parallel_group_ranks_with_cp = []
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
    
    return all_data_parallel_group_ranks_with_cp

def generate_context_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    """
    Generate data parallel groups considering context parallelism.
    """
    assert world_size % (pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size) == 0, "world_size must be divisible by the product of pipeline_model_parallel_size, tensor_model_parallel_size, and context_parallel_size"
    all_data_parallel_group_ranks_with_cp = []
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
    
    return all_data_parallel_group_ranks_with_cp

def generate_tensor_model_parallel_groups(world_size, tensor_model_parallel_size):
    """
    Generate model parallel groups based on tensor model parallel size.
    """
    assert world_size % tensor_model_parallel_size == 0, "world_size must be divisible by tensor_model_parallel_size"
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    tensor_model_parallel_group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        tensor_model_parallel_group_ranks.append(list(ranks))
    return tensor_model_parallel_group_ranks

def generate_pipeline_parallel_groups(world_size, pipeline_model_parallel_size):
    """
    Generate pipeline parallel groups based on pipeline model parallel size.
    """
    assert world_size % pipeline_model_parallel_size == 0, "world_size must be divisible by pipeline_model_parallel_size"
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    pipline_parallel_group_ranks = []

    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        pipline_parallel_group_ranks.append(list(ranks))
    return pipline_parallel_group_ranks

def generate_context_parallel_groups(world_size, context_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size):
    """
    Generate context parallel groups based on context parallel size, considering tensor and pipeline model parallel sizes.
    """
    assert world_size % (context_parallel_size * tensor_model_parallel_size * pipeline_model_parallel_size) == 0, "world_size must be divisible by the product of context_parallel_size, tensor_model_parallel_size, and pipeline_model_parallel_size"
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
    context_parallel_group_ranks = []
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

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
    return context_parallel_group_ranks

def plot_parallel_groups(title="Parallel Groups", dp_groups=None, tp_groups=None, pp_groups=None, cp_groups=None):
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define the spacing between blocks and their size
    block_size = 700  # Size of the blocks in the scatter plot
    spacing = 1.5  # Spacing multiplier between blocks
    if cp_groups is None:
        cp_offset_x = 0
        cp_offset_y = 0
        tp_offset_x = 0.2
        tp_offset_y = -0.2
        if tp_groups:
            pp_offset_x = 0.4
            pp_offset_y = -0.4
        else:
            pp_offset_x = 0.2
            pp_offset_y = -0.2
    else:
        cp_offset_x = 0.2
        cp_offset_y = -0.2
        tp_offset_x = 0.4
        tp_offset_y = -0.4
        if tp_groups:
            pp_offset_x = 0.6
            pp_offset_y = -0.6
        else:
            pp_offset_x = 0.4
            pp_offset_y = -0.4

    # Adjust the grid layout to map GPU ranks from top-left to bottom-right
    num_cols = 4  # Number of columns in the grid
    x_positions = np.tile(np.arange(num_cols), num_cols) * spacing
    y_positions = np.repeat(np.arange(num_cols), num_cols)[::-1] * spacing  # Reverse to start from top

    dp_colors = plt.cm.tab20(np.linspace(0, 1, len(dp_groups)))

    # 使用tab20b提高颜色区分度
    if tp_groups is not None:
        tp_colors = plt.cm.tab20b(np.linspace(0, 1, len(tp_groups)))

    # 如果需要更多颜色，可以考虑结合使用tab20b和tab20c
    if pp_groups is not None:
        pp_colors = plt.cm.tab20c(np.linspace(0, 1, len(pp_groups)))

    if cp_groups is not None:
        cp_colors = plt.cm.tab20c(np.linspace(0, 1, len(cp_groups)))

    if cp_groups is not None:
        for group_idx, group in enumerate(cp_groups):
            for rank in group:
                x = x_positions[rank % (num_cols*num_cols)] + cp_offset_x
                y = y_positions[rank % (num_cols*num_cols)] + cp_offset_y
                ax.scatter(x, y, s=block_size, color=cp_colors[group_idx], edgecolor='black', zorder=5, marker='s')
                ax.text(x, y, f'CP{rank}', ha='center', va='center', color='white', fontsize=8, zorder=6, fontweight='bold')
    
    for group_idx, group in enumerate(dp_groups):
        for rank in group:
            x = x_positions[rank % (num_cols*num_cols)]
            y = y_positions[rank % (num_cols*num_cols)]
            ax.scatter(x, y, s=block_size, color=dp_colors[group_idx], edgecolor='black', zorder=5, marker='>')
            ax.text(x, y, f'DP{rank}', ha='center', va='center', color='white', fontsize=8, zorder=6, fontweight='bold')
    
    if tp_groups is not None:
        for group_idx, group in enumerate(tp_groups):
            for rank in group:
                x = x_positions[rank % (num_cols*num_cols)] + tp_offset_x
                y = y_positions[rank % (num_cols*num_cols)] + tp_offset_y
                ax.scatter(x, y, s=block_size, color=tp_colors[group_idx], edgecolor='black', zorder=5, marker='p')
                ax.text(x, y, f'TP{rank}', ha='center', va='center', color='white', fontsize=8, zorder=6, fontweight='bold')

    if pp_groups is not None:
        for group_idx, group in enumerate(pp_groups):
            for rank in group:
                x = x_positions[rank % (num_cols*num_cols)] + pp_offset_x
                y = y_positions[rank % (num_cols*num_cols)] + pp_offset_y
                ax.scatter(x, y, s=block_size, color=pp_colors[group_idx], edgecolor='black', zorder=5, marker='h')
                ax.text(x, y, f'PP{rank}', ha='center', va='center', color='white', fontsize=8, zorder=6, fontweight='bold')

    # Draw a separating line between Node0 and Node1
    mid_y_position = np.max(y_positions) / 2
    ax.axhline(y=mid_y_position, color='black', linestyle='-', linewidth=2, zorder=0)

    # Add Node labels
    ax.text(-spacing, max(y_positions)/4, 'Node1', verticalalignment='center', fontsize=12)
    ax.text(-spacing, 3*max(y_positions)/4, 'Node0', verticalalignment='center', fontsize=12)
    
    # Adjusting the appearance
    ax.set_aspect('equal')  # Keep the aspect ratio square
    ax.axis('off')  # Turn off the axis
    plt.title(title, pad=30)

    return fig

# Gradio interface setup
def create_interface():
    def update_plot(parallel_group_type, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size, unused_text):
        world_size = 16  # Fixed world size for 2 machines with 8 GPUs each
        
        description = descriptions.get(parallel_group_type, "Invalid parallel group type")
        
        # Initialize groups to None
        data_groups = tp_groups = pp_groups = cp_groups = None

        if "CP" in parallel_group_type or parallel_group_type == 'Context Parallel':
            cp_groups = generate_context_parallel_groups(world_size, context_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size)
            if "DP" in parallel_group_type:
                data_groups = generate_context_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
        else:
            if "DP" in parallel_group_type or parallel_group_type == 'Data Parallel':
                data_groups = generate_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
        
        if parallel_group_type in ['Tensor Model Parallel', 'DP+TP', 'DP+TP+PP', 'CP+DP+TP', 'CP+DP+TP+PP']:
            tp_groups = generate_tensor_model_parallel_groups(world_size, tensor_model_parallel_size)
        if parallel_group_type in ['Pipeline Parallel', 'DP+PP', 'DP+TP+PP', 'CP+DP+PP', 'CP+DP+TP+PP']:
            pp_groups = generate_pipeline_parallel_groups(world_size, pipeline_model_parallel_size)

        # Prepare text description for display
        groups_list_str = ""
        if data_groups:
            groups_list_str += "Data Parallel Groups:\n"
            groups_list_str += "\n".join([f"Data Group {idx + 1}: {group}" for idx, group in enumerate(data_groups)])
            groups_list_str += "\n--------------------------------------\n"
        if tp_groups:
            groups_list_str += "Tensor Model Parallel Groups:\n"
            groups_list_str += "\n".join([f"Tensor Group {idx + 1}: {group}" for idx, group in enumerate(tp_groups)])
            groups_list_str += "\n--------------------------------------\n"
        if pp_groups:
            groups_list_str += "Pipeline Model Parallel Groups:\n"
            groups_list_str += "\n".join([f"Pipeline Group {idx + 1}: {group}" for idx, group in enumerate(pp_groups)])
            groups_list_str += "\n--------------------------------------\n"
        if cp_groups:
            groups_list_str += "Context Parallel Groups:\n"
            groups_list_str += "\n".join([f"Context Group {idx + 1}: {group}" for idx, group in enumerate(cp_groups)])
            groups_list_str += "\n--------------------------------------\n"

        text_to_display = f"==========Parallel Groups Display==========\n\n{groups_list_str}\n\n{description}"

        # Generate the figure with the parallel groups
        fig = plot_parallel_groups(f"{parallel_group_type} Groups", data_groups if data_groups else [], tp_groups=tp_groups, pp_groups=pp_groups, cp_groups=cp_groups)
        
        return fig, text_to_display

    iface = gr.Interface(
        fn=update_plot,
        inputs=[
            gr.Dropdown(['Data Parallel', 'Tensor Model Parallel', 'Pipeline Parallel', 'Context Parallel',
                         'DP+TP', 'DP+PP', 'DP+TP+PP',
                         'CP+DP', 'CP+DP+TP', 'CP+DP+PP', 'CP+DP+TP+PP'], label="Parallel Group Type"),
            gr.Slider(1, 8, step=1, label="Tensor Model Parallel Size"),
            gr.Slider(1, 8, step=1, label="Pipeline Model Parallel Size"),
            gr.Slider(1, 8, step=1, label="Context Parallel Size"),
            gr.Textbox(basic_texts, interactive=False)
        ],
        outputs=[
            "plot",
            "text"
        ],
        title="Megatron-LM Parallel Group Visualization",
        description="Select parallel sizes and types to visualize different parallel groups with distinct colors. This includes combinations of Data Parallel (DP), Tensor Model Parallel (TP), Pipeline Parallel (PP), and Context Parallel (CP). Note that the size of data parallelism is automatically calculated based on world_size (which is stable at 16 here) as well as tensor_model_parallel_size, pipeline_model_parallel_size, and context_parallel_size.",
        live=True
    )
    
    return iface

# Create and launch the interface
iface = create_interface()
iface.launch(share=False)

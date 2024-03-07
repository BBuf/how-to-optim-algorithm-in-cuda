import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def generate_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    """
    Generate data parallel groups based on the provided parallelism parameters.
    """

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

def generate_tensor_model_parallel_groups(world_size, tensor_model_parallel_size):
    """
    Generate model parallel groups based on tensor model parallel size.
    """
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

def plot_parallel_groups(groups, title="Parallel Groups"):
    """
    Plot the parallel groups with different colors for each group, with increased spacing between blocks,
    and a line separating Node0 and Node1, with labels.
    """
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define the spacing between blocks and their size
    block_size = 700  # Size of the blocks in the scatter plot
    spacing = 1.5  # Spacing multiplier between blocks
    
    # Adjust the grid layout to map GPU ranks from top-left to bottom-right
    num_cols = 4  # Number of columns in the grid
    x_positions = np.tile(np.arange(num_cols), num_cols) * spacing
    y_positions = np.repeat(np.arange(num_cols), num_cols)[::-1] * spacing  # Reverse to start from top

    # Plot each group with a unique color
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))  # Generate distinct colors
    for group_idx, group in enumerate(groups):
        for rank in group:
            # Calculate position based on the rank's order in a sequential layout
            x = x_positions[rank % (num_cols*num_cols)]
            y = y_positions[rank % (num_cols*num_cols)]
            ax.scatter(x, y, s=block_size, color=colors[group_idx], edgecolor='black', zorder=5)
            # Add GPU label inside the block
            ax.text(x, y, f'GPU{rank}', ha='center', va='center', color='white', fontsize=8, zorder=6, fontweight='bold')

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
    def update_plot(parallel_group_type, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
        world_size = 16  # Fixed world size for 2 machines with 8 GPUs each
        
        if parallel_group_type == 'Data Parallel':
            groups = generate_data_parallel_groups(world_size, tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
        elif parallel_group_type == 'Tensor Model Parallel':
            groups = generate_tensor_model_parallel_groups(world_size, tensor_model_parallel_size)
        elif parallel_group_type == 'Pipeline Parallel':
            groups = generate_pipeline_parallel_groups(world_size, pipeline_model_parallel_size)
        elif parallel_group_type == 'Context Parallel':
            groups = generate_context_parallel_groups(world_size, context_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size)
        else:
            raise ValueError('Invalid parallel group type')
        fig = plot_parallel_groups(groups, f"{parallel_group_type} Groups")
        return fig

    iface = gr.Interface(
        fn=update_plot,
        inputs=[
            gr.Dropdown(['Data Parallel', 'Tensor Model Parallel', 'Pipeline Parallel', 'Context Parallel'], label="Parallel Group Type"),
            gr.Slider(1, 8, step=1, label="Tensor Model Parallel Size"),
            gr.Slider(1, 8, step=1, label="Pipeline Model Parallel Size"),
            gr.Slider(1, 8, step=1, label="Context Parallel Size"),
        ],
        outputs="plot",
        title="Megatron-LM Parallel Group Visualization",
        description="Select parallel sizes and types to visualize different parallel groups with distinct colors. Note that The size of data parallelism is automatically calculated based on world_size (which is stable at 16 here) as well as tensor_model_parallel_size, pipeline_model_parallel_size, and context_parallel_size.",
        live=True
    )
    
    return iface

# Create and launch the interface
iface = create_interface()
iface.launch()

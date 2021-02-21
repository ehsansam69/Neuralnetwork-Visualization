import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # noqa:E402
from PIL import Image #noqa:E402

#choose a color pallete
BLUE = "#04253a"
GREEN = '#4C837a'
TAN = '#e1ddbf'
DPI = 300

#This in an image will be used
FILLER_IMAGE_FILENAME = "pic7.jpg"

"""
Generate a auto encoder neural network visualization
"""
#changing this adjust the size and layout of the visualization
number_of_layers = int(input("How many layers? "))
N_NODES_BY_LAYER = []
for layer in range(number_of_layers):
    a = int(input("number of node in layer?"))
    N_NODES_BY_LAYER.append(a)

FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
RIGHT_BORDER = 0.7
LEFT_BORDER = 0.7
TOP_BORDER =0.8
BOTTOM_BORDER =0.6

N_IMAGE_PIXEL_COLS = 64
N_IMAGE_PIXEL_ROWS =48
#N_NODES_BY_LAYER =[10,7,5,8]

INPUT_IMAGE_BOTTOM = 5
INPUT_IMAGE_HEIGHT =0.25*FIGURE_HEIGHT
ERROR_IMAGE_SCALE =0.7
ERROR_GAP_SCALE =0.3
BETWEEN_LAYER_SCALE =0.8
BETWEEN_NODE_SCALE = 0.4

def main():
   p = construct_parameters()
   fig,ax_boss = create_background(p)
   p = find_node_image_size(p)
   p = find_gap_between_layer(p)
   p = find_gap_between_node(p)
   p = find_error_image_position(p)


   filler_image = load_filler_image()
   image_axes =[]
   add_input_image(fig,image_axes, p,filler_image)
   for i_layer in range(p['network']['n_layers']):
       add_node_images(fig,i_layer,image_axes,p,filler_image)
   add_output_image(fig,image_axes, p,filler_image)
   add_error_image(fig,image_axes, p,filler_image)
   add_layer_connection(ax_boss, image_axes)
   save_nn_viz(fig, postfix="test.png")
   print(image_axes)

   print("parameters: ")
   for key,value in p.items():
       print(key ," : ",value)


def construct_parameters():
    """
    Build a dictionary of parameters that describe the size and location of the elements of the visualization
    this is a convenient way to pass the collection of them around
    """
    #enforce square pixels. each pixel will have the same height and width
    aspect_ratio = N_IMAGE_PIXEL_COLS/N_IMAGE_PIXEL_ROWS

    parameters ={}
    parameters['figure'] = {
        'height':FIGURE_HEIGHT,
        'width': FIGURE_WIDTH,
    }
    parameters['input'] = {
        'n_cols':N_IMAGE_PIXEL_COLS,
        'n_rows': N_IMAGE_PIXEL_ROWS,
        'aspect_ratio' : aspect_ratio,
        'image':{
            'bottom': INPUT_IMAGE_BOTTOM,
            'height':INPUT_IMAGE_HEIGHT,
            'width': INPUT_IMAGE_HEIGHT*aspect_ratio
        }
    }
    #the network as a whole
    parameters['network'] ={
        'n_nodes': N_NODES_BY_LAYER,
        'n_layers':len(N_NODES_BY_LAYER),
        'max_nodes':np.max(N_NODES_BY_LAYER)
    }

    #individual node image

    parameters['node_image']={
        'height':0,
        'width':0
    }

    parameters['error_image'] = {
        'left':0,
        'right':0,
        'width':parameters['input']['image']['width']*ERROR_IMAGE_SCALE,
        'height':parameters['input']['image']['height']*ERROR_IMAGE_SCALE
    }

    parameters['gap']={
        'right_border':RIGHT_BORDER,
        'left_border':LEFT_BORDER,
        'bottom_border':BOTTOM_BORDER,
        'top_border':TOP_BORDER,
        'between_layer':0,
        'between_layer_scale':BETWEEN_LAYER_SCALE,
        'between_node':0,
        'between_node_scale':BETWEEN_NODE_SCALE,
        'error_gap_scale':ERROR_GAP_SCALE
    }



    return parameters

def create_background(p):
    fig = plt.figure(
        edgecolor=TAN,
        facecolor=GREEN,
        figsize=(p['figure']['width'],p['figure']['height']),
        linewidth = 4,
    )

    ax_boss = fig.add_axes((0,0,1,1),facecolor = 'none')
    ax_boss.set_xlim(0,1)
    ax_boss.set_ylim(0, 1)

    return fig,ax_boss
def find_node_image_size(p):
    total_space_to_fill= (
        p['figure']['height']
        -p['gap']['bottom_border']
        -p['gap']['top_border']
    )

    height_contrained_by_height = (
        total_space_to_fill/(
        p['network']['max_nodes']
        + (p['network']['max_nodes']-1)*p['gap']['between_node_scale']
    )
    )
    total_space_to_fill = (
            p['figure']['width']
            - p['gap']['left_border']
            - p['gap']['right_border']
            - 2*p['input']['image']['width']
    )
    width_contrained_by_width = (
            total_space_to_fill / (
            p['network']['n_layers']
            + (p['network']['n_layers'] + 1) * p['gap']['between_layer_scale'])
    )
    height_contrained_by_width = (
    width_contrained_by_width
    / p['input']['aspect_ratio']
    )

    #see which contraint is more restrictive and go with it
    p['node_image']['height'] = np.minimum(
        height_contrained_by_height,
        height_contrained_by_width)
    p['node_image']['width'] = p['node_image']['height']*p['input']['aspect_ratio']
    return p
def find_gap_between_layer(p):
    horizontal_gap_total = (
        p['figure']['width']
        - 2*p['input']['image']['width']
        -p['network']['n_layers']*p['node_image']['width']
        -p['gap']['left_border']
        -p['gap']['right_border']
    )
    n_horizontal_gaps = p['network']['n_layers']+1
    p['gap']['between_layer'] = horizontal_gap_total / n_horizontal_gaps
    return p
def find_gap_between_node(p):
    vertical_gap_total = (
        p['figure']['height']
        - p['gap']['top_border']
        - p['gap']['bottom_border']
        - p['network']['max_nodes']*p['node_image']['height']
    )
    n_vertical_gaps = p['network']['max_nodes'] - 1
    p['gap']['between_node'] = vertical_gap_total / n_vertical_gaps
    return p
def find_error_image_position(p):
    """
    where exactly should the error image be positioned
    """
    p['error_image']['bottom'] = (
        p['input']['image']['bottom']
        -p['input']['image']['height']
        *p['gap']['error_gap_scale']
        -p['error_image']['height']
    )
    error_image_center = (
        p['figure']['width']
        -p['gap']['right_border']
        -p['input']['image']['width'] /2
    )
    p['error_image']['left'] = (
        error_image_center - p['error_image']['width']/2
    )
    return p
def add_input_image(fig,image_axes, p,filler_image):
    """
    All axes to be added use the rectangle specification
    (left,bottom,width,height)
    """
    absolute_pos = (
        p['gap']['left_border'],
        p['input']['image']['bottom'],
        p['input']['image']['width'],
        p['input']['image']['height']
    )
    ax_input = add_image_axes(fig,p,absolute_pos)
    add_filler_image(
        ax_input,
        p['input']['n_rows'],
        p['input']['n_cols'],
        filler_image,
    )
    image_axes.append([ax_input])
def add_image_axes(fig,p,absolute_pos):
    scaled_pos = (
        absolute_pos[0] / p['figure']['width'],
        absolute_pos[1] / p['figure']['height'],
        absolute_pos[2] / p['figure']['width'],
        absolute_pos[3] / p['figure']['height'],
    )
    ax= fig.add_axes(scaled_pos)
    ax.tick_params(bottom=False, top=False, right=False, left=False)
    ax.tick_params(
        labelbottom=False, labeltop=False, labelright=False, labelleft=False
    )
    ax.spines['top'].set_color(TAN)
    ax.spines['bottom'].set_color(TAN)
    ax.spines['left'].set_color(TAN)
    ax.spines['right'].set_color(TAN)
    return ax
# def add_filler_image(ax,n_im_rows,n_im_cols):  #random filler
#     """
#     add chunk of images as a place holder.
#     """fill_patch = np.random.sample(size=(n_im_rows,n_im_cols))
#     ax.imshow(fill_patch, cmap='inferno')
def add_node_images(fig,i_layer,image_axes,p,filler_image):
    node_image_left = (
        p['gap']['left_border']
        +p['input']['image']['width']
        + i_layer *p['node_image']['width']
        +(i_layer+1)*p['gap']['between_layer']
    )
    n_nodes = p['network']['n_nodes'][i_layer]
    total_layer_height = (
        n_nodes*p['node_image']['height']
        +(n_nodes-1)*p['gap']['between_node']
    )

    layer_bottom = (p['figure']['height'] - total_layer_height)/2
    layer_axes =[]

    for i_node in range(n_nodes):
        node_image_bottom = (layer_bottom + i_node *(
            p['node_image']['height'] + p['gap']['between_node'])
        )
        absolute_pos = (
            node_image_left,
            node_image_bottom,
            p['node_image']['width'],
            p['node_image']['height']
        )
        ax = add_image_axes(fig, p, absolute_pos)
        add_filler_image(
            ax,
            p['input']['n_rows'],
            p['input']['n_cols'],
            filler_image
        )
        layer_axes.append(ax)
    image_axes.append(layer_axes)
def add_output_image(fig,image_axes,p,filler_image):
    output_image_left = (
        p['figure']['width']
        -p['input']['image']['width']
        -p['gap']['right_border']
    )
    absolute_pos = (
        output_image_left,
        p['input']['image']['bottom'],
        p['input']['image']['width'],
        p['input']['image']['height']
    )
    ax_output = add_image_axes(fig,p,absolute_pos)
    add_filler_image(
        ax_output,
        p['input']['n_rows'],
        p['input']['n_cols'],
        filler_image
    )
    image_axes.append([ax_output])
def add_error_image(fig,image_axes, p,filler_image):
    absolute_pos = (
        p['error_image']['left'],
        p['error_image']['bottom'],
        p['error_image']['width'],
        p['error_image']['height'],
    )
    ax_error = add_image_axes(fig, p, absolute_pos)
    add_filler_image(
        ax_error,
        p['input']['n_rows'],
        p['input']['n_cols'],
        filler_image
    )
def load_filler_image():
    img = Image.open(FILLER_IMAGE_FILENAME)
    img.load()
    color_img = np.asarray(img, dtype="int32")
    #average the three color channels together to create a monochrom image
    bw_img = np.mean(color_img,axis=2, dtype="int32")
    return bw_img
def add_filler_image(ax,n_im_rows,n_im_cols,filler_image):
    #add a chunk of image as a place holder
    top = np.random.randint(filler_image.shape[0] - n_im_rows)
    left = np.random.randint(filler_image.shape[1] - n_im_cols)
    bottom = top +n_im_rows
    right = left + n_im_cols
    fill_patch = filler_image[top:bottom,left:right]
    ax.imshow(fill_patch,cmap="inferno")
def add_layer_connection(ax_boss, image_axes):
    """
    Add in the connectors between all layers
    Treat the input image as the first layer and output layer as last
    """
    for i_start_layer in range(len(image_axes)-1):
        n_start_nodes = len(image_axes[i_start_layer])
        n_end_nodes = len(image_axes[i_start_layer +1])
        x_start = image_axes[i_start_layer][0].get_position().x1
        x_end = image_axes[i_start_layer+1][0].get_position().x0

        for i_start_ax , ax_start in enumerate(image_axes[i_start_layer]):
            ax_start_pos = ax_start.get_position()
            y_start_min = ax_start_pos.y0
            y_start_max = ax_start_pos.y1
            y_start = (y_start_max + y_start_min) / 2
            start_spacing = (y_start_max - y_start_min)/(n_end_nodes+1)

            for i_end_ax, ax_end in enumerate(image_axes[i_start_layer + 1]):
                ax_end_pos = ax_end.get_position()
                y_end_min = ax_end_pos.y0
                y_end_max = ax_end_pos.y1
                y_end = (y_end_max + y_end_min) / 2
                end_spacing = (y_end_max -y_end_min) / (n_start_nodes + 1)

                #spread out y_start and y_end a bit
                x = [x_start, x_end]
                y = [y_start_min+start_spacing*(i_end_ax+1), y_end_min+end_spacing*(i_start_ax +1)]
                plot_connection(ax_boss, x_start,x_end,y_start,y_end)
def plot_connection(ax_boss, x0, x1, y0, y1):
    # for curvy connection
    weight = np.random.sample() *2 -1
    x = np.linspace(x0, x1, num = 50)
    y = y0 + (y1 - y0) * (-np.cos(np.pi * (x - x0)/(x1 - x0))+ 1)/2
    if weight>0:
        conn_color = TAN
    else:
        conn_color = BLUE
    ax_boss.plot(x, y, color = conn_color, linewidth = 3 *weight)


def save_nn_viz(fig, postfix ="0"):
    #generate a new filename for each step
    base_name = 'nn_viz_'
    filename = base_name +postfix+".png"
    fig.savefig(
        filename,
    edgecolor = fig.get_edgecolor(),
    facecolor = fig.get_facecolor(),
    dpi = DPI)

if __name__ == "__main__":
    main()


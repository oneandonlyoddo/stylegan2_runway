import runway
#from runway.data_types import vector, image
import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib
import numpy as np
import PIL.Image
import pickle

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    #path = './models/Heather-Day-6000.pkl'
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        #print('Loading networks from "%s"...' % path)
        #_G, _D, Gs = pretrained_networks.load_networks(path)
        G, D, Gs = pickle.load(file)
    return Gs
    
generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

outputs={
    'image': runway.image
}

@runway.command('generate', inputs=generate_inputs, outputs=outputs, description='Generate an image.')
def generate(model, input_args):
    
    noise_vars = [var for name, var in model.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = input_args['truncation']
    rnd = np.random.RandomState(0)
    z = input_args['z']
    #print(z)
    latents = z.reshape((1, 512))
    #print(z)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    image = model.run(latents, None, **Gs_kwargs) # [minibatch, height, width, channel]
    output = PIL.Image.fromarray(image[0], 'RGB')
    return {'image': output}

if __name__ == '__main__':
    runway.run(debug=True)

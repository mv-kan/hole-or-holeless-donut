from duckduckgo_search import DDGS
from fastcore.all import *
from fastai.vision.widgets import *
from matplotlib import pyplot

def search_images(term, max_images=80):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(term)
        count = 0
        ddgs_images_list = []
        while count < max_images:
            image = next(ddgs_images_gen)
            ddgs_images_list.append(image.get('image'))
            count = count+1
        return ddgs_images_list

if __name__ == "__main__":
    #NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
    #    If you get a JSON error, just try running it again (it may take a couple of tries).
    urls = search_images('donut photos', max_images=1)
    urls[0]
    from fastdownload import download_url
    dest = 'hole.jpg'
    download_url(urls[0], dest, show_progress=False)

    from fastai.vision.all import *
    im = Image.open(dest)
    im.to_thumb(256,256)
    download_url(search_images('filled donut photo', max_images=1)[0], 'filled.jpg', show_progress=False)
    Image.open('filled.jpg').to_thumb(256,256)
    
    searches = 'donut-photo','filled-donut-photo'
    path = Path('hole-or-filled')
    from time import sleep

    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o}'))
        sleep(10)
        resize_images(path/o, max_size=400, dest=path/o)

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(len(failed))

    dls = DataBlock(
        # batch_tfms=aug_transforms(mult=1),
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)
    
    dls.train.show_batch(max_n=8, nrows=2, unique=True)
    
    pyplot.show()
    dls.show_batch(max_n=6)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(10)

    p = 'hole.jpg'
    is_what,_,probs = learn.predict(PILImage.create(p))
    print(f"{p} is a: {is_what}.")

    p = 'filled.jpg'
    is_what,_,probs = learn.predict(PILImage.create(p))
    print(f"{p} is a: {is_what}.")
    learn.export("hole-or-filled.pkl")

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(5, nrows=1)

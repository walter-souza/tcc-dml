import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products
# import datasets.garcom
import datasets.biblept
import datasets.biblept_acf


def select(dataset, opt, data_path, tokenizer=None):
    if 'cub200' in dataset:
        return cub200.Give(opt, data_path)

    if 'cars196' in dataset:
        return cars196.Give(opt, data_path)

    if 'online_products' in dataset:
        return stanford_online_products.Give(opt, data_path)

    # if 'garcom' in dataset:
    #     return garcom.Give(opt, 'garcom_data', tokenizer)

    if 'biblept' in dataset:
        file = 'bible_pt.xml'
        books = ['JHN', 'MAT']
        return biblept.Give(opt, file, tokenizer, books)

    if 'bible_acf' in dataset:
        file = 'pt_acf.json'
        books = ['jo', 'mt']
        return biblept_acf.Give(opt, file, tokenizer, books)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & online_products!'.format(dataset))

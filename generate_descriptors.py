import os
from openai import OpenAI
import json
import pandas as pd

import itertools

from descriptor_strings import stringtolist

import api_secrets
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

def generate_region_prompt(category_name: str, region_name: str = 'Africa'):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return [
        {"role": "system", "content": f"""Q: What are useful visual features for distinguishing a lemur in a photo?
        A: There are several useful visual features to tell there is a lemur in a photo:
        - four-limbed primate
        - black, grey, white, brown, or red-brown
        - wet and hairless nose with curved nostrils
        - long tail
        - large eyes
        - furry bodies
        - clawed hands and feet

        Q: What are useful visual features for distinguishing a television in a photo?
        A: There are several useful visual features to tell there is a television in a photo:
        - electronic device
        - black or grey
        - a large, rectangular screen
        - a stand or mount to support the screen
        - one or more speakers
        - a power cord
        - input ports for connecting to other devices
        - a remote control \n \n Please directly output answers."""},
        {"role": "user", "content": f"""Q: What are useful features for distinguishing a {category_name} in {region_name} in a photo?
        A: There are several useful visual features to tell there is a {category_name} in {region_name} in a photo:
        -
        """}
    ]

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    # prompts = [generate_region_prompt(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    

def obtain_geode_descriptors_and_save(filename, class_list, region_name='Africa'):
    responses = []
    descriptors = {}
    descriptors_list = []
    
    
    prompts = [generate_region_prompt(category.replace('_', ' '), region_name) for category in class_list]
    
    # for category in class_list:
    #     prompt = generate_region_prompt(category.replace('_', ' '), region_name)
    #     response_text = client.chat.completions.create(
    #         model="gpt-4",
    #         messages=prompt
    #     ).choices[0].message.content
    #     descriptors_list.append(stringtolist(response_text))
    #     print(response_text)
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [client.chat.completions.create(
            model="gpt-4",
            messages=prompt
        )  for prompt_partition in partition(prompts, 20) for prompt in prompt_partition]
    response_texts = [r.choices[0].message.content for r in responses]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # check if there is empty list as values or values as single element list in descriptors
    # if so, query the model again for the missing descriptors
    while any([not value or len(value) == 1 for value in descriptors.values()]):
        key = next(key for key, value in descriptors.items() if not value or len(value) == 1)
        print(key)
        prompt = generate_region_prompt(key.replace('_', ' '), region_name)
        response_text = client.chat.completions.create(
            model="gpt-4",
            messages=prompt
        ).choices[0].message.content
        descriptors[key] = stringtolist(response_text)
        print(response_text)
    # for key, value in descriptors.items():
    #     if not value:
    #         prompt = generate_region_prompt(key.replace('_', ' '), region_name)
    #         response_text = client.chat.completions.create(
    #             model="gpt-4",
    #             messages=prompt
    #         ).choices[0].message.content
    #         descriptors[key] = stringtolist(response_text)

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp, indent=4)


df = pd.read_csv('/local2/data/xuanming/geode/index.csv', index_col=False)
class_list = list(df['object'].unique())  # class list in Geo-DE
obtain_geode_descriptors_and_save('descriptors_geode_westasia', class_list, 'West Asia')
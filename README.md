# llm-transformers

[![PyPI](https://img.shields.io/pypi/v/llm-transformers.svg)](https://pypi.org/project/llm-transformers/)
[![Changelog](https://img.shields.io/github/v/release/rectalogic/llm-transformers?include_prereleases&label=changelog)](https://github.com/rectalogic/llm-transformers/releases)
[![Tests](https://github.com/rectalogic/llm-transformers/actions/workflows/test.yml/badge.svg)](https://github.com/rectalogic/llm-transformers/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/rectalogic/llm-transformers/blob/main/LICENSE)

Plugin for llm adding support for [🤗 Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-transformers
```
## Usage

XXX document `-o verbose True`

## Transformer tasks

### [audio-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AudioClassificationPipeline)

The `audio-classification` task takes an audio URL or path, for example:
```sh-session
$ llm -m transformers -o task audio-classification https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
_unknown_ (0.9972336888313293)
left (0.0019911774434149265)
yes (0.0003051063104066998)
down (0.0002108386834152043)
stop (0.00011406492558307946)
```

### [automatic-speech-recognition](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)

```sh-session
$ llm -m transformers -o task automatic-speech-recognition https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOWER FAT AND SAUCE
```

### [depth-estimation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.DepthEstimationPipeline)

The `depth-estimation` task accepts an image url or path as input and generates an image file as output:
```sh-session
$ llm -m transformers -o task depth-estimation http://images.cocodataset.org/val2017/000000039769.jpg
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpjvp9uo7x.png
```
XXX embed image here?

### [document-question-answering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline)

The `document-question-answering` task requires a `context` option which is a file or URL to an image:

```sh-session
$ llm -m transformers -o task document-question-answering -o context https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png "What is the invoice number?"
us-001
```

### [feature-extraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
Not supported.

### [fill-mask](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FillMaskPipeline)

`fill-mask` requires a placeholder in the prompt, thiis is typically `<mask>` but is different for different models:

```sh-session
$ llm -m transformers -o task fill-mask "My <mask> is about to explode"
My brain is about to explode (score=0.09140042215585709)
My heart is about to explode (score=0.07742168009281158)
My head is about to explode (score=0.05137857422232628)
My fridge is about to explode (score=0.029346412047743797)
My house is about to explode (score=0.02866862528026104)
```

### [image-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageClassificationPipeline)

```sh-session
$ llm -m transformers -o task image-classification https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
macaw (0.9905233979225159)
African grey, African gray, Psittacus erithacus (0.005603480152785778)
toucan (0.001056905253790319)
sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita (0.0006811501225456595)
lorikeet (0.0006714339251630008)
```

### [image-feature-extraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageFeatureExtractionPipeline)
Not supported.

### [image-segmentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageSegmentationPipeline)

```sh-session
$ llm -m transformers -o task image-segmentation https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmp0z8zvd8i.png (bird: 0.999439)
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpik_7r5qn.png (bird: 0.998787)
```

### [image-to-image](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToImagePipeline)

```sh-session
$ llm -m transformers -o task image-to-image http://images.cocodataset.org/val2017/000000039769.jpg
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpczogz6cb.png
```

### [image-to-text](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToTextPipeline)

```sh-session
$ llm -m transformers -o task image-to-text https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
two birds are standing next to each other
```

### [mask-generation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.MaskGenerationPipeline)
Not supported.

### [object-detection](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ObjectDetectionPipeline)

```sh-session
$ llm -m transformers -o task object-detection https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
[
    {
        "score": 0.9966394901275635,
        "label": "bird",
        "box": {
            "xmin": 69,
            "ymin": 171,
            "xmax": 396,
            "ymax": 507
        }
    },
    {
        "score": 0.999381422996521,
        "label": "bird",
        "box": {
            "xmin": 398,
            "ymin": 105,
            "xmax": 767,
            "ymax": 507
        }
    }
]
```

### [question-answering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline)

```sh-session
$ llm -m transformers -o task question-answering -o context "My name is Wolfgang and I live in Berlin" "Where do I live?"
Berlin
```

### [summarization](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.SummarizationPipeline)

Specify additional pipeline keyword args with the `kwargs` model option, a JSON text document:
```sh-session
$ llm -m transformers -o task summarization "An apple a day, keeps the doctor away"
 An apple a day, keeps the doctor away from your doctor away . An apple every day is an apple that keeps you from going to the doctor . The apple is the best way to keep your doctor from getting a doctor's orders, according to the author of The Daily Mail
$ llm -m transformers -o task summarization -o kwargs '{"min_length": 2, "max_length": 7}' "An apple a day, keeps the doctor away"
 An apple a day
```

### [table-question-answering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TableQuestionAnsweringPipeline)

`table-question-answering` takes a required `context` option - a path to a CSV file.

```sh-session
$ cat <<EOF > /tmp/t.csv
> Repository,Stars,Contributors,Programming language
Transformers,36542,651,Python
Datasets,4512,77,Python
Tokenizers,3934,34,"Rust, Python and NodeJS"
> EOF
$ llm -m transformers -o task table-question-answering -o context /tmp/t.csv "How many stars does the transformers repository have?"
AVERAGE > 36542
$ llm -m transformers -o task table-question-answering -o context /tmp/t.csv "How many contributors do all Python language repositories have?"
SUM > 651, 77
```

### [text2text-generation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline)

```sh-session
$ llm -m transformers -o task text2text-generation "question: What is 42 ? context: 42 is the answer to life, the universe and everything"
the answer to life, the universe and everything
```

### [text-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline)

```sh-session
$ llm -m transformers -o task text-classification "We are very happy to show you the 🤗 Transformers library"
POSITIVE (0.9997681975364685)
```

### [text-generation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline)

Some `text-generation` models can be chatted with.

```sh-session
$ llm -m transformers -o task text-generation "I am going to elect"
I am going to elect the president of Mexico and that president should vote for our president," he said. "That's not very popular. That's not the American way. I would not want voters to accept the fact that that guy's running a
$ llm -m transformers -o task text-generation -o model HuggingFaceH4/zephyr-7b-beta -o kwargs '{"max_new_tokens": 2}' "What is the capital of France? Answer in one word."
Paris
$ llm chat -m transformers -o task text-generation -o model HuggingFaceH4/zephyr-7b-beta -o kwargs '{"max_new_tokens": 25}'
Chatting with transformers
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> What is the capital of France?
The capital of France is Paris (French: Paris). The official name of the city is "Ville de Paris"
> What question did I just ask you?
Your question was: "What is the capital of France?"
> quit
```

### [text-to-audio](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextToAudioPipeline)

`text-to-audio` generates audio, the response is the path to the audio file.
```sh-session
$ llm -m transformers -o kwargs '{"generate_kwargs": {"max_new_tokens": 100}}' -o model facebook/musicgen-small "techno music"
/var/folders/b1/1j9kkk053txc5krqbh0lj5t00000gn/T/tmpoueh05y6.wav
```

### [token-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)

```sh-session
$ llm -m transformers -o task token-classification "My name is Sarah and I live in London"
Sarah (I-PER: 0.9982994198799133)
London (I-LOC: 0.998397171497345)
```

### [translation_xx_to_yy](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TranslationPipeline)

Substitute the from and to language codes into the task name, e.g. from `en` to `fr` would use task `translation_en_to_fr`:

```sh-session
$ llm -m transformers -o task translation_en_to_fr "How old are you?"
 quel âge êtes-vous?
```

### [video-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.VideoClassificationPipeline)

`video-classification` task expects a video path or URL as the prompt:

```sh-session
$ llm -m transformers -o task video-classification https://huggingface.co/datasets/Xuehai/MMWorld/resolve/main/Amazing%20street%20dance%20performance%20from%20Futunity%20UK%20-%20Move%20It%202013/Amazing%20street%20dance%20performance%20from%20Futunity%20UK%20-%20Move%20It%202013.mp4
dancing ballet (0.006608937866985798)
spinning poi (0.006111182738095522)
air drumming (0.005756791681051254)
singing (0.005747966933995485)
punching bag (0.00565463537350297)
```

### [visual-question-answering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.VisualQuestionAnsweringPipeline)

`visual-question-answering` task requires an `context` option - a file or URL to an image:

```sh-session
$ llm -m transformers -o task visual-question-answering -o context https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png "What is she wearing?"
hat (0.9480269551277161)
fedora (0.00863664224743843)
clothes (0.003124270820990205)
sun hat (0.002937435172498226)
nothing (0.0020962499547749758)
```

### [zero-shot-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline)

`zero-shot-classification` requires a comma separated list of labels to be specified in the `context` model option:

```sh-session
$ llm -m transformers -o task zero-shot-classification -o context "urgent,not urgent,phone,tablet,computer" "I have a problem with my iphone that needs to be resolved asap!!"
urgent (0.5036348700523376)
phone (0.4788002371788025)
computer (0.012600351125001907)
not urgent (0.0026557915844023228)
tablet (0.0023087668232619762)
```

### [zero-shot-image-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotImageClassificationPipeline)

`zero-shot-image-classification` requires a comma separated list of labels to be specified in the `context` model option. The prompt is a path or URL to an image:

```sh-session
$ llm -m transformers -o task zero-shot-image-classification -o context "black and white,photorealist,painting" https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png
black and white (0.9736384749412537)
photorealist (0.02141517587006092)
painting (0.004946451168507338)
```

### [zero-shot-audio-classification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotAudioClassificationPipeline)

`zero-shot-audio-classification` requires a comma separated list of labels to be specified in the `context` model option. The prompt is a path or URL to an audio:

```sh-session
$ llm -m transformers -o task zero-shot-audio-classification -o context "Sound of a bird,Sound of a dog" https://huggingface.co/datasets/s3prl/Nonspeech/resolve/main/animal_sound/n52.wav
Sound of a bird (0.9998763799667358)
Sound of a dog (0.00012355657236184925)
```

### [zero-shot-object-detection](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotObjectDetectionPipeline)

`zero-shot-object-detection` requires a comma separated list of labels to be specified in the `context` model option. The prompt is a path or URL to an image.
The response is JSON and includes a bounding box for each label:

```sh-session
$ llm -m transformers -o task zero-shot-object-detection -o context "cat,couch" http://images.cocodataset.org/val2017/000000039769.jpg
[
    {
        "score": 0.2868139445781708,
        "label": "cat",
        "box": {
            "xmin": 324,
            "ymin": 20,
            "xmax": 640,
            "ymax": 373
        }
    },
    {
        "score": 0.2537268102169037,
        "label": "cat",
        "box": {
            "xmin": 1,
            "ymin": 55,
            "xmax": 315,
            "ymax": 472
        }
    },
    {
        "score": 0.12082991003990173,
        "label": "couch",
        "box": {
            "xmin": 4,
            "ymin": 0,
            "xmax": 642,
            "ymax": 476
        }
    }
]
```


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-transformers
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

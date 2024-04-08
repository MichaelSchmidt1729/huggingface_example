from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")

def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

text_to_summarize = """
The Biological Cosmic Ray Experiment (BIOCORE) was a lunar science experiment, that flew on board Command module America as part of Apollo 17.[1] The goal of the BIOCORE experiment was to explore whether or not high energy cosmic rays produced visibly identifiable trauma to brain and eye tissues.[2][3][4]

Background
During the Apollo 11 mission, Buzz Aldrin and Neil Armstrong reported observing flashes of light whenever they had their eyes closed or when the inside of their spacecraft was dimly lit.[3][5] Every Apollo mission subsequently reported the exact same phenomenon.[5] It was surmised that the cause of these light emissions were the result of heavy-ion cosmic rays interacting with the light detecting cells in the retina of a human eye.[3] This led scientists to question whether or not such heavy cosmic rays might cause damage to the tissues that form the eye, brain and other organs as they transit through a human body.[3]

To assess this, mice with plastic dosimeters surgically implanted under their scalps were flown aboard Apollo 17. The aim for the experiment was to assess whether any microscopic lesions could be visibly identified within the brain, eye and other organ tissues of the mice and whether they could be attributed to high-energy cosmic rays.[2]

"""
summary = generate_summary(text_to_summarize)
print(summary)

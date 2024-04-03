import torch
from transformers import pipeline

summarizer = pipeline('summarization')

wikipedia_text = """Die Triodos Bank N.V. ist eine Aktiengesellschaft niederländischen Rechts mit Hauptsitz in Zeist, Niederlande.[5] Sie umfasst Zweigniederlassungen und Repräsentanzen in sechs europäischen Ländern,[4] sowie die Tochterunternehmen Triodos Investment Management und Triodos Private Banking.[6] Zusammen bietet die Triodos Gruppe ein breites Spektrum an Dienstleistungen an: Bankdienstleistungen, Fondsmanagement, Projektentwicklung, Depotgeschäft, Beteiligungskapital, Unternehmensfinanzierung und Private Banking.[7][8]
Zum 31. Dezember 2015 betrug das Geschäftsvolumen der Triodos Bankengruppe rund 12,3 Milliarden Euro,[4] sie vergab über 44.000 (2015) Kredite[4] und betreute in Europa über 700.000 Kundenkonten.[4]
"""
result = summarizer(wikipedia_text, max_length=150, min_length=30)

result[0]["summary_text"]

sentiment = pipeline('sentiment-analysis')

text = ('I loved it', 'I hated the service','Es war so lala')
sentiment_result = sentiment(text)

for r, t in zip(sentiment_result, text):
  print(t, '-->', r['label'])

translator = pipeline('translation', model="Helsinki-NLP/opus-mt-en-fr")

translator(result[0]['summary_text '])


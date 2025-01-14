# DigitalZibaldone (DZ) Entity Linking Dataset

## Description
This dataset has been derived from [Digital Zibaldone](https://digitalzibaldone.net/) the digital edition of 
*Zibaldone* by Giacomo Leopardi. This dataset include notes written by Leopardi in 1821 (p.1000 to p.2000) and in 1823 (p.2700 to p. 3000). It includes annotations that reference people, locations, and works, linked where possible to entities in Wikidata. When an entity is not linked to Wikidata, VIAF identifiers are provided wherever possible. This dataset is designed for Named Entity Recognition and Entity Linking 
tasks, facilitating the identification and association of text references with entities in a knowledge base.

## Sampling process

The dataset was sampled by selecting two sections of the Zibaldone written in separate years: one section from p. 1000 to p. 2000 written in 1821; another section from p.2700 to p. 4000 written in 1823.
To ensure data quality and maintain a balanced distribution, paragraphs with an anomalous word count relative to a 
normal distribution were excluded. Also paragraphs without references to named entities were excluded. In order to 
divide dataset in training, development and testing, the data was split with a 70/15/15 ratio using a stratified 
sampling strategy based on the creation date of the documents.

## Entity Tagset

List of annotated entities (coarse level only):

* Person (PER)
* Location (LOC)
* Literary Works (WORK)

## Dataset Statistics

### Training Set

| Statistic                   | Value |
|-----------------------------|-------|
| Number of documents         | 735   |
| Total number of annotations | 2,935 |
| Annotations for persons     | 1,661 |
| Annotations for locations   | 488   |
| Annotations for works       | 786   |
| NIL annotations             | 158   |

### Development Set

| Statistic                   | Value |
|-----------------------------|-------|
| Number of documents         | 157   |
| Total number of annotations | 727   |
| Annotations for persons     | 375   |
| Annotations for locations   | 149   |
| Annotations for works       | 203   |
| NIL annotations             | 57    |

### Test Set

| Statistic                   | Value |
|-----------------------------|-------|
| Number of documents         | 158   |
| Total number of annotations | 617   |
| Annotations for persons     | 318   |
| Annotations for locations   | 130   |
| Annotations for works       | 169   |
| NIL annotations             | 38    |

## Creation Date
December 12, 2024

## Purpose and Use
The **DigitalZibaldone Entity Linking Dataset** serves as a resource for training and evaluating entity linking models, particularly in historical texts. The presence of NIL annotations highlights instances where entities could not be found in existing knowledge bases, challenging models to handle out-of-knowledge-base (OOV) entities.

---

**Note**: This dataset provides a rich environment for testing entity linking systems within historical literary texts, where certain references may lack modern or conventional equivalents in current knowledge graphs like Wikidata.

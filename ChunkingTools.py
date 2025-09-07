from langchain.docstore.document import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import RootModel, BaseModel, Field
from rapidfuzz import fuzz
from typing import *
import rich
import os, sys
from config import init_LLM, get_llm_info
from LogWriter import LogWriter

'''
This file contains custom Document splitter, focus on agentic chuncking and splitting
'''

# ===============================
# StandAloneFactTextSplitter

# -------------------------------
# stand alone sentence extraction

# modify from prompt in https://arxiv.org/pdf/2312.06648.pdf
proposal_template ='''
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]
'''
proposal_indexing_template = ChatPromptTemplate.from_messages([ SystemMessage(proposal_template), HumanMessagePromptTemplate.from_template('''Decompose the following:\n{input}''')])

# Create Structured Extraction
class Sentences(BaseModel):
    sentences: List[str]

class StandAloneSentencesExtractor:
    def __init__(self, llm: ChatOllama):
        ''' Extract Document text into Stand Alone Sentences '''
        self.structured_llm = llm.with_structured_output(Sentences).bind(options={"temperature": 0.0})
        self.token_limit = int(get_llm_info(llm).get('Model', None).get('context length'))
        self.token_limit = round(self.token_limit * 0.95) if self.token_limit else None
        self.llm_chunking_runnable = proposal_indexing_template | self.structured_llm

    def split_text(self, text:str, chunk_size:int, chunk_overlap:int) -> tuple[ List[List[str]], List[str] ]:
        ''' Two stage text split for long documents, returns extracted facts and paragraphs. First stage execute Recursive Charater Splitter. Second stage execute fact extraction. '''
        rough_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        paragraphs = rough_splitter.split_text(text)
        extract_facts = []
        for p in paragraphs:
            facts = self.extract_fact(p)
            extract_facts.append(facts)
        return extract_facts, paragraphs

    def extract_fact(self, paragraphs:str) -> List[Optional[str]]:
        ''' list out some facts in paragraphs using llm, recommand spilt long text into shorter paragraphs before execution '''
        if self.token_limit:
            expected_token = count_tokens_approximately(paragraphs)
            assert expected_token <= self.token_limit, f"Maximum input token reached. {expected_token} Tokens"
        chunks = self.llm_chunking_runnable.invoke({"input" : paragraphs}).model_dump()
        return chunks.get("sentences", [])

# ===============================
# Claim and Citation Extraction

claim_extract_prompt = '''
You are a teacher preparing learning material for student, right now you are reading through a PARAGRAPHS form an article.
Tell me few points of summary what the PARAGRAPHS said and which sentences are related to your summary.
PARAGRAPHS expected to be a little bit unstructured, but still holds valuable informations.

Here are some keyword you should know:
    The word PARAGRAPHS means the small part of a larger article you are reading and summarizing.
    The word SINGLE_CLAIM_AND_CITATIONS means a single claim that is supported by one or more citations from reading PARAGRAPHS.
    The word ALL_CLAIMS_WITH_CITATIONS means a list that contains alot of SINGLE_CLAIM_AND_CITATIONS that has been summarized from PARAGRAPHS.
    The word OUTPUT_EXAMPLE means the output format you should follow.

Here we discribe what kind of context you should write in your summzrization:
    SUMMARIZED_CLAIM inside of SINGLE_CLAIM_AND_CITATIONS are stand alone sentences that says a fact or a conclusion supported by many citations.
    EXACT_RELATED_CITATION_SENTENCES inside of SINGLE_CLAIM_AND_CITATIONS are exactly same matched parts in PARAGRAPHS that supports a SUMMARIZED_CLAIM in SINGLE_CLAIM_AND_CITATIONS.
    EXACT_RELATED_CITATION_SENTENCES should be exact same match that can be search in PARAGRAPHS, you should never use any symbol to shorten citation sentence.
    SUMMARIZED_CLAIM could have one or more EXACT_RELATED_CITATION_SENTENCES supported, but multiple SUMMARIZED_CLAIM can have shared EXACT_RELATED_CITATION_SENTENCES support.
    You should not write anything else other then format of OUTPUT_EXAMPLE, no beautify sentences allowed.

Your final goal is to read through PARAGRAPHS to summarize into many SUMMARIZED_CLAIM and find EXACT_RELATED_CITATION_SENTENCES that supports SUMMARIZED_CLAIM, output EXACT_RELATED_CITATION_SENTENCES and SUMMARIZED_CLAIM with the format of OUTPUT_EXAMPLE.

If PARAGRAPHS can not summarize into any vaild claims, you should write an MEANLESS_PARAGRAPHS_FILLER_OUTPUT to point out a meanless PARAGRAPHS as your conclusion, do not make up a fake claim from your own knowledge.
MEANLESS_PARAGRAPHS_FILLER_OUTPUT:
    ALL_CLAIMS_WITH_CITATIONS([
        SINGLE_CLAIM_AND_CITATIONS("chunk is unrecognizable", []),
    ])

With all the goals, rules and keyword explained, here is a OUTPUT_EXAMPLE for output format.

OUTPUT_EXAMPLE:
    ALL_CLAIMS_WITH_CITATIONS([
        SINGLE_CLAIM_AND_CITATIONS(
            SUMMARIZED_CLAIM,
            EXACT_RELATED_CITATION_SENTENCES=[
                <CITATION>, 
                ],
            ),
        SINGLE_CLAIM_AND_CITATIONS(
            SUMMARIZED_CLAIM,
            EXACT_RELATED_CITATION_SENTENCES=[
                <CITATION>, 
                <CITATION>, 
                <CITATION>, 
                ],
            ),
        SINGLE_CLAIM_AND_CITATIONS(
            SUMMARIZED_CLAIM,
            EXACT_RELATED_CITATION_SENTENCES=[
                <CITATION>, 
                <CITATION>, 
                <CITATION>, 
                <CITATION>, 
                ...
                ],
            ),
        ...
    ])

Remeber, give as much citation as you can, it will help you write a better summarization.
Now that I explain all the rules, keywords and output format you should know, lets start reading through the PARAGRAPHS.

PARAGRAPHS:{input}
'''

claim_extract_template = ChatPromptTemplate.from_messages([
    ("human", claim_extract_prompt)
])

class SingleClaimAndCitations(BaseModel):
    ''' Structure guide for llm extraction '''
    summarized_claim: str
    exact_related_citation_sentences: List[str]

class AllClaimsWithCitations(RootModel[List[SingleClaimAndCitations]]):
    ''' Structure guide for llm extraction '''
    pass

claim_extract_failed = AllClaimsWithCitations(root=[
    SingleClaimAndCitations(summarized_claim="CLAIM_EXTRACT_FAILED", exact_related_citation_sentences=[])
    ])

class DetailedClaim(BaseModel):
    ''' llm generated claims with citation checked '''
    claim: str
    real_citations: List[Tuple[float, str]] = Field(default_factory=list)
    fake_citations: List[Tuple[float, str]] = Field(default_factory=list)

class Claim(BaseModel):
    ''' claim structure for final result output '''
    metadata: Dict[Literal["document", "page", "chunk_id"], str] = Field(default={"document" : "unknown document", "page" : "0", "chunk_id" : "0"})
    claim: str
    citations: List[Tuple[float, str]] = Field(default_factory=list)

class ClaimWithCitationsExtractor:
    def __init__(self, llm: ChatOllama):
        ''' Extract Document text into Stand Alone Sentences '''
        self.structured_llm = llm.with_structured_output(AllClaimsWithCitations)
        self.structured_llm = self.structured_llm.bind(options={
            "temperature": 0.0,
            "format":"json",
            "repeat_last_n": 300,
            "num_predict": 1000,
            "max_retries": 3
            }, 
            stop=["\n\n\n", "```"]
            )
        self.extract_progress: float = 1 # 0.0~1.0
        self.llm_chunking_runnable = proposal_indexing_template | self.structured_llm

    def split_text(self, text:str, chunk_size:int, chunk_overlap:int, citation_score:float=0.95, logger:Optional[LogWriter]=None) -> Tuple[ List[ List[Claim] ], List[ Tuple[int, str] ] ]:
        ''' 
        Extract claims with supported citations.
        Two stage extraction for any document:
        1. Recursive Splitter for chunking.
        2. LLM claim extraction for claim and citations.
        '''
        #
        rough_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = rough_splitter.split_text(text)
        rich.print(f"Processing {len(chunks)} chunks")
        self.extract_progress = 0 if len(chunks) else 1
        #
        chunk_claims   : List[ List[Claim] ]     = []
        labeled_chunks : List[ Tuple[int, str] ] = []
        for chunk_id, chunk in enumerate(chunks, start=1):
            d_claims: List[DetailedClaim] = self.extract_claim(chunk, citation_score)
            logger.write_log(log_message=chunk,  message_section=f"Chunk {chunk_id}") if logger else None
            citation_count_msg = '\n'.join([f"Chunk {chunk_id :<3} Claim {c_num :<3} Citation [ Real {len(c.real_citations) :<3} Fake {len(c.fake_citations) :<3} ]" for c_num, c in enumerate(d_claims)])
            rich.print(citation_count_msg)
            logger.write_log(log_message=citation_count_msg,  message_section=f"Chunk {chunk_id} citation count") if logger else None
            logger.write_log(log_message=d_claims, message_section=f"Chunk {chunk_id} Claims") if logger else None
            claims = []
            for c in d_claims:
                _c = Claim(claim=c.claim, citations=c.real_citations)
                _c.metadata.update({"chunk_id" : str(chunk_id)})
                claims.append(_c)
            chunk_claims.append(claims)
            labeled_chunks.append((chunk_id, chunk))
            self.extract_progress = chunk_id / len(chunks)
        self.extract_progress = 1
        return chunk_claims, chunks
    
    def check_citation(self, chunk: str, citations: List[str], fuzzy_score:float=0.95) -> Dict[Literal["real", "fake"], List[ Tuple[float, str] ] ]:
        ''' Match citations in chunk string, returns citations actually exists '''
        fuzzy_score = max(min(fuzzy_score, 1), 0)
        match_scoring = lambda needle, hay: round(fuzz.partial_ratio(str(needle), str(hay)) / 100.0, 4)
        real_c, fake_c = [], []
        for c in citations:
            s = match_scoring(c, chunk)
            real_c.append((s, c)) if s >= fuzzy_score else fake_c.append((s, c))
        return {"real" : real_c, "fake" : fake_c}

    def extract_claim(self, chunk:str, citation_score:float=0.95) -> List[Optional[DetailedClaim]]:
        ''' Extract claims and citation from given chunk by llm, label claim with given chunk_id '''
        llm_extract_claims: SingleClaimAndCitations = self.llm_chunking_runnable.with_fallbacks([RunnableLambda(lambda _: claim_extract_failed)]).invoke({"input" : chunk})
        llm_extract_claims: List[SingleClaimAndCitations] = llm_extract_claims.root
        all_new_claims = []
        for llm_c in llm_extract_claims:
            claim     = llm_c.summarized_claim or ""
            citations = list(set(llm_c.exact_related_citation_sentences)) or []
            verified_citations = self.check_citation(chunk, citations, citation_score)
            new_claim = DetailedClaim(
                claim = claim,
                real_citations = verified_citations["real"],
                fake_citations = verified_citations["fake"],
            )
            all_new_claims.append(new_claim)
        return all_new_claims

# ===============================
# Test Code

if __name__ == "__main__":
    test_text = '''
Source: https://thetravelblog.at/exploring-taiwan-travel-guide-for-first-time-visitors
Author: Marion Vicenta Payr
Time: 18. February 2025

First time in Taiwan?
I was lucky enough to join a trip together with a group of journalists and a wonderful Taiwanase travel guide for a week. Of course this is too short to become a Taiwan expert myself, but it was just long enough to get a first glimpse of the country (well, and to leave me longing for more). There’s a lot to take away from this week and that I learned, hence I’m sharing this guide with those of you who are also planning their first trip to Taiwan.

In this guide I’ll share some of my tips if you’re also planning a first time visit of Taiwan.

You’ll learn what you can expect (and not expect) from a destination influenced heavily by Chinese and Japanese immigration and traditions mixed with the indigenous heritage.

Streets of Taipei, TaiwanStreets of Taipei, TaiwanStreets of Taipei, TaiwanTaipei 101, TaiwanView down from elephant mountain on Taipei, Taiwan

11 Things to know before visiting Taiwan
Before sharing our itinerary and tips for places to visit and things to do, let me share a few things, that you should know before you go. To be completely frank, I didn’t know too much about Taiwan before planning my first trip. In our western media the country usually only makes appearances when it comes to political issues with China (note that Taiwan is still officially called “Republic of China”, clearly showing that the complicated history of the country is still reflected in today’s politics). Yet, when it comes to anything beyond these discussions we rarely read too much about Taiwan in our media.

So, here’s what I learned and think is important to know before a trip:

Green spaces: Taiwan has a population density of approximately 650 people per square kilometer, making it one of the most densely populated regions globally (ranked 17th in the world). Despite this, around 60% of Taiwan’s land area is covered in forests and green spaces, which surprised me – I hadn’t expected so much nature before I went. That’s why Taiwan is also famous for it’s national parks and lots of hiking opportunities.
Vegetarian food: On the plane to Taipei I watched a documentary about the night markets of Taiwan and in that show they only featured one vegetarian dish, alongside over 10 meat heavy snacks. So I was unsure how many options I would find as a vegetarian. Yet later I found out that Taiwan also has a significant vegetarian population (an estimated 13% of Taiwanese people), largely due to religious and cultural influences. So, in the end I found vegetarian food everywhere. It helps to learn how to say that you’re looking for vegetarian food in Chinese (“Zhe shi su de ma?” which means “Is this vegetarian?”). Another Tip: Vegetarian food is often marked with these symbols: 卍 (Buddhist swastika) and 素 (vegetarian) character.
Night markets: We tried a few local restaurants together with our guide, which all served decent food. But except for two very famous restaurants I personally preferred the food at the night markets and I now understand the hype. They are truly an experience and even if you’re vegetarian and not ready to opt for chicken butts or blood sausage you can find some really incredible foods. My personal favourites were the countless king oyster mushroom variations as well as fried scallion fingers and peanut ice cream wraps with fresh cilantro (!). More on that later, but just make sure you plan enough night market visits during your stay.
Bubble tea: Taiwan not only has an incredible tea culture (you can also visit tea farms and learn about the differences between green tea, oloong and black tea and their fermentation process), but it’s also famous as the birthplace of bubble tea. I had the most amazing bubble tea variations here and you cannot miss sampling this iconic drink.
Cash is still king: Speaking of night markets. While most shops, hotels and restaurants accept credit cards, many night markets and small vendors only take cash, so it’s wise to carry some New Taiwan Dollars (NTD) with you. I picked up about 100 USD worth of NTD (around 3,000 NTD) straight at the airport and that went a long way. Night market food is relatively cheap (think 120 NTD for a bubble tea or 150 NTD for a large portion of grilled mushrooms).
Convenience stores: 7-Eleven, FamilyMart, and other convenience stores are ubiquitous and offer services like topping up your EasyCard (to use public transport) or purchasing mobile data (I opted for Chunghwa Telecom and paid around 20 USD for unlimited data for a week). There’s also cooking stations with brownish-coloured tea eggs (a delicacy you have to try!), and are perfect for grabbing a quick oolong tea or onigiri.
Scooters everywhere: Like in many Asian cities scooters dominate the streets. Be cautious when crossing roads, especially since there’s now many e-scooters silently gliding through the streets.
Hot springs culture: Taiwan has numerous hot springs, thanks to its volcanic geology. We visited Beitou in Taipei, which is a popular spot among tourists. Opposed to Japanese onsen baths you are allowed to enter with tattoos, but bathing caps are mandatory (and will be provided alongside sandals and Kimonos if you visit one of the fancier hot springs).
Temples galore: Temples are abundant, even in the smallest villages you’ll find more than one temple or shrine. Often they represent the mixed beliefs of the Taiwanese, incorporating local folkloristic beliefs into Buddhist and Taoist religions. Tip: Always enter on the right side and leave on the left to make sure to follow local customs.
Weather variations: Taiwan’s weather varies greatly by region and season. In February we had sunny hot days where we walked around in t-shirts and then a day later we were wrapped in wool sweaters and light raincoats (and very glad we brought our umbrellas). And although the island isn’t that large, the subtropical North can be cooler than the tropical South. Typhoon season (June–September) might not be the best time to visit, but more below in the “Best time to visit” segment.
No tipping: This is something to note and adapt, as we are not used to this at all. But tipping is not common in Taiwan (except for tour guides). It will be considered offensive if you try to tip against these standards, so keep that change (and use it at a food stall later).
Wenwu temple at Sun moon lake in TaiwanWenwu temple at Sun moon lake in TaiwanWenwu temple at Sun moon lake in Taiwan

Best time to visit Taiwan
The best time to visit Taiwan is during the fall (October to November), winter (December to February), or spring (March to May). Summer tends to be hot and humid making it less ideal to visit. Fall offers clear skies, comfortable temperatures, and colourful autumn scenery. Winter, especially December through February, is a great time to visit with mild temperatures in most regions. Winter also brings the festive spirit of the vibrant Lunar New Year celebrations (which we had just missed by a few days).

If you want to come during Chinese new year here are the dates for the next years to mark in your calendar:

February 17 , 2026
February 6, 2027
January 26, 2028
Personally we visited in Mid February, which marked some of the first cherry blossoms (note that the further South you go the earlier these start to bloom, in some places already in January). But the main reason for our visit was the celebration of the Lantern Festival (always on the 15th day of the new year according to the Chinese calendar), when glowing lanterns light up the night sky. The weather in February was warmer than we had expected and it truly felt like a warm spring already.

Sakura: in February cherry blossoms in Taipei, TaiwanSakura: in February cherry blossoms in Taipei, TaiwanSpring in TaiwanSakura: Cherry blooms in February in many places in TaiwanXuan Zang Temple at Sun moon lake, Taiwan

Our one week itinerary & travel tips
We only stayed a (way too short) week in Taiwan, offering a first glimpse of the diversity of cultures and sceneries. If you also only have one week we would recommend to combine a few days in the city of Taipei with one destination in nature.

We stayed two nights at the famous Sun Moon Lake in central Taiwan, a turquoise coloured gem lined with bamboo forests and mountain peaks. We could’ve easily extended this by a few more days to have the chance to explore some of the hikes on offer, but Taipei was calling and we also enjoyed our time in the city (think night markets, temple visits, museums and of course a visit of the landmark Taipei 101).

Fu-ren Temple in Daxi, TaiwanTemple in Nantou County, TaiwanFu-ren Temple in Daxi, Taiwan

Tips for Sun Moon lake
Many people come to Taiwan for the vibrant city life, but as many of you know I really enjoy nature and was happy we also got a chance to visit the largest lake in Taiwan: Sun Moon lake in Nantou County in central Taiwan.

Before we dive into our tips, let’s get one thing out of the way first: Swimming is generally not allowed here (except for two days per year, where tens of thousands flock to the lake for this special opportunity). Therefore a visit here is more about boat & bike rides, mountain hikes and enjoying the scenery around the lake.

Boat ride at Sun moon lake in TaiwanBike ride at Sun moon lake in TaiwanBoat ride at Sun moon lake in Taiwan

We stayed at Ita Thao village on the Eastern shore of the lake at the super modern Wyndham Sun Moon Lake hotel. The highlight of the hotel was that they discovered a hot spring during the excavation works and decided to embrace this opportunity to transform the hotel and equip each room with traditional hot spring stone baths. Perfect after a day of lake explorations if you ask me (just don’t heat it up all the way to 42 degrees like I did, 38-39 degrees might be enough).

Don’t miss visiting Wenfu temple and Ci’en pagoda once you’re here as well as renting bikes to ride along one of the hundred most beautiful bike paths in the world (according to National Geographic Traveler). We rented our e-bikes from GIANT – Sun Moon Lake Station and made our way to the organically shaped concrete Xiangshan Visitor Center and then continued to Xiangshan Scenic Outlook and all the way to the end of the path at the “看見拉魯觀景台 / Observation deck”.

Ci'en pagoda at Sun moon lake in TaiwanCi'en pagoda at Sun moon lake in TaiwanSun moon lake in Taiwan

At Shuishe Pier you can hop on a boat to see Lalu island up close (not too interesting, but the folkloristic stories about it are fascinating) and then stop for a visit of Xuan Zang Temple – dedicated to the Chinese Buddhist master translator who wandered all of India in search of the original Sanskrit texts. Unfortunately we didn’t have a chance to ride the gondola in Ita Thao, but that would definitely be on my list for a next visit.

In the village of Ita Thao there’s a small night market, which is worth a visit to grab some snacks in the evening or sample the traditional millet wine, that the indigenous Thao people invented.

Ita Thao village at Sun moon lake in Taiwan

Tips for Taipei
If you’re visiting Taipei the night markets are an absolute must – and there’s plenty of choice. We stayed at the famous Palais de Chine hotel, so Ningxia Night market was our closest option. Although we only arrived at 9pm the first evening, we still hopped over and I found the most delicious grilled king oyster mushrooms in a soy marinade here, which I topped with lemon juice and spicy curry powder (the stand isn’t marked in Google Maps, but it’s in the arcades on the left side opposite of stand #53, that sells Exploded Egg yolk Taro balls, but which I didn’t like too much).

Palais de Chine hotel in Taipei, TaiwanPalais de Chine hotel in Taipei, TaiwanPalais de Chine hotel in Taipei, TaiwanPalais de Chine hotel in Taipei, TaiwanPalais de Chine hotel in Taipei, Taiwan

Speaking about food I want to recommend my favourite restaurant here in Taipei, and no, it’s not the famous Din Tai Fung (although the dumplings here are really good). But my personal favourite must’ve been Yang Shin Vegetarian, especially for their “Signature Spicy Sichuan Wontons” and the “Beijing Duck Style King Oyster Mushroom”. So so so good!

Yang Shin Vegetarian restaurant in Taipei, TaiwanYang Shin Vegetarian restaurant in Taipei, TaiwanYang Shin Vegetarian restaurant in Taipei, Taiwan

Of course we couldn’t resist another night market visit, which was Shilin Night market, and about 10x larger than the little Ningxia market. The vibe here is incredibly lively and a lot of the vendors here speak English as it’s more touristy too. My personal favourite here was the “Scallion Fried dough Stick” and the dessert! A peanut and taro ice cream wrap (in a type of crepe made from rice dough) filled with shaved peanuts and loads of fresh cilantro. Never had anything like this before and I absolutely loved it. Make sure to get it from this guy in the picture as he’s the one who invented it (now many stands copied the dish).

Shilin night market in Taipei, TaiwanPeanut and taro ice cream wrap with cilantro at Shilin night market in Taipei, TaiwanShilin night market in Taipei, Taiwan

Other must do’s include visiting the observation deck at Taipei 101, the National Palace Museum and of course – you have to hike up the Elephant Mountain Trail (we went for sunrise, which didn’t really happen, but we were almost alone – sunset is supposed to be the best in terms of views, but maybe gets a bit crowded). They say it’s 600 steps to the top. I didn’t count, but it was enough to give me sore muscles the next day 😉

Elephant mountain trail in Taipei, TaiwanElephant mountain trail in Taipei, TaiwanElephant mountain viewpoint in Taipei, Taiwan

We also stopped at Chiang Kai-shek Memorial Hall, the perfect place to see the cherry trees in bloom in February! Many locals also come here for birdwatching, so if you’re into that bring your long tele photo lenses or binoculars – you can see water hens and egrets and even king fishers at the pond in the park.

Chiang Kai-shek Memorial Hall in Taipei, TaiwanChiang Kai-shek Memorial Hall in Taipei, TaiwanChiang Kai-shek Memorial Hall in Taipei, TaiwanChiang Kai-shek Memorial Hall in Taipei, TaiwanChiang Kai-shek Memorial Hall in Taipei, Taiwan

One afternoon was dedicated to the hot springs at Beitou and a visit of the lovely (albeit small) Beitou museum, housed in a former Ryokan style Japanese home and dedicated to indigenous art and an exhibition about the bathing culture in Taiwan.

Beitou museum in Taipei, TaiwanBeitou museum in Taipei, TaiwanBeitou museum in Taipei, TaiwanBeitou museum in Taipei, TaiwanBeitou in Taipei, Taiwan

What to skip: Shifen (Pingxi) sky lanterns
Now here comes an unpopular opinion, but personally I would skip a visit of Shifen (Pingxi). I know the sky lantern releasing here is on top of everyone’s must do list for Taiwan, but I found it an incredibly overwhelming and overrated mass tourism experience.

Imagine a tiny village overrun by thousands of tourists everyday, who all assemble in the same street to release lanterns in the sky. You’ll get shoved through that road to then stand in line until you get to release your lantern (not before having your photos taken of course).

Minutes later the lantern disappears in the distance only to land in the forests around the village. Supposedly locals are getting paid to recover the trashed lanterns and bring them to the landfill, but we didn’t see that. What we did see is dozens of lantern everywhere in the trees and the riverbed next to the village.

My least favourite place: Shifen (Pingxi) in TaiwanMy least favourite place: Shifen (Pingxi) in Taiwan, where they release sky lanternsMy least favourite place: Shifen (Pingxi) in Taiwan, where they release sky lanterns

Visit the lantern festival instead
Instead of an unsustainable Shifen visit I suggest to visit Taiwan during the yearly lantern festival celebration. This festival takes place at a different city every year and showcases light installations, that aren’t turned into trash within minutes.

This year’s lantern festival was held in Taoyuan and featured a sustainability section with light installations made from bamboo as well as a light show at the main snake-shaped lantern (to celebrate the year of the snake).

Taoyuan Lantern festival in Taiwan 2025Taoyuan Lantern festival in Taiwan 2025Taoyuan Lantern festival in Taiwan 2025

What about Jiufen?
Of course we also visited the famous Jiufen, which left me with mixed feelings. Like many tourists we only came during a day trip and were shoved through the main street alongside masses of other visitors that day. I do understand the hype in some ways, as the sloped mountain village must’ve been scenic a while ago, but now it all feels a bit like Disneyworld (or Hallstatt for that matter).

The main street is lined with shops, souvenir stalls, tea houses and restaurants, so it’s a very commercialised experience. We made the best of it and simply embraced the shopping, got some souvenirs (like a personalised stamp made from wood and carved intarsia) and bubble tea. I feel if you were to stay a night it might be a different experience as most visitors will leave in the late afternoon, but at the same time I don’t know if it really is worth the visit.

Next time I would definitely opt for a village that is more off-the-beaten path instead.

Jiufen TaiwanJiufen TaiwanJiufen Taiwan

Take-aways from a week in Taiwan
This week was merely a first glimpse of the country. Now I know a tiny fragment of Taiwan, but with many reasons to return for more. Here’s what’s on my list for a next visit:

I would love to visit a serpentine quarry to learn more about the famous green snake stone.
I want to hike the countless mountain trails and spot more monkeys, for example at Kenting National Park.
I want to learn about the fermentation of soy beans – for example at May-dong Traditional Handmade Sauces in central Taiwan’s Taichung city.
I want to stay at Hoshinoya Guguan to experinence the hot spring town of Guguan.
And of course I want to visit the incredible Dragon and Tiger Pagodas and Sanfeng Temple in Kaohsiung.
I also want to visit the historical city of Tainan, the oldest city in the country.
And last but not least I’d love to travel to Alishan National Scenic Area for a longer nature escape.
There’s a lot more to see and do in Taiwan, so I hope to be back one day! Let me know if I missed something that you’ve experienced in Taiwan and I’ll add it to my bucket list.

View from elephant mountain trail in Taipei, Taiwan

Pin and save this post if you’re planning a Taiwan trip:

Disclaimer: This is not a sponsored post. We were invited to discover Taiwan with the Taiwan Tourism Authority during a press trip, but are not obligated to write about it on this blog. The views in this blogpost are our own.
Beginner's guide to TaiwanTaiwan for first time visitorsTaiwan Travel Guide'''

    logger = LogWriter("ChunkToolTest", "test_log")
    logger.clear()

    llm = init_LLM()
    claim_extractor = ClaimWithCitationsExtractor(llm)
    logger.write_s_line(0)
    logger.write_log(log_message="Start", message_section="Claim With Citations Extractor")
    chunk_claims, labeled_chunks = claim_extractor.split_text(test_text, 800, 200, logger=logger)
    logger.write_s_line(0)

    splitter = StandAloneSentencesExtractor(llm)
    facts, paragraphs = splitter.split_text(test_text, 800, 200)
    logger.write_s_line(0)
    logger.write_log(log_message="Start", message_section="Stand Alone Sentences Extractor")
    for num, fact_paragraph in enumerate(zip(facts, paragraphs)):
        fact, paragraph = fact_paragraph
        logger.write_s_line(1)
        logger.write_log(log_message=paragraph, message_section=f"Paragraph {num}")
        logger.write_log(log_message=fact, message_section=f"Facts {num}")
        logger.write_s_line(1)
    logger.write_s_line(0)
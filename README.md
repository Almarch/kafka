# Kafka-trained LLM

<img alt="Llama" src="https://github.com/user-attachments/assets/91f06e0b-7c79-4de9-9386-8dab581f8289" width="400px" align="right"/>

The goal of this project is to fine-tune a LLM on French litterature, and especially to imprint it with a single book which is the French translation of Kafka's novel: the Castle. The author never ended this book, so the idea is to prolongate it with new chapters.

It is designed for a Nvidia GPU with at least 12 Go VRAM. The [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is needed. A minimal 32Go RAM is safe for the data preparation step.

It takes as input:

- a free-to-operate base model: [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1), a 1.1B parameters with a 2048 tokens context window.
- a French copyleft translation of The Castle by Kafka, available [here](https://ekladata.com/QAPtMO27HuI4V0hLEhOUd3sv0Nw/Kafka-Le-Chateau.pdf).
- a French literature dataset: [Gallica](https://huggingface.co/datasets/PleIAs/French-PD-Books) for French language enrichment and narrative consistency.

## 🐳 Launch

Build & launch the docker container:

```sh
docker build -t kafka .
docker run -d \
  --name kafka \
  -v ./project:/project \
  -p 8000:8000 \
  --gpus all \
  kafka
```

The project is supported by a notebook, available at [port 8000](http://localhost:8000) once the container runs. This notebook may be used to monitor the training and test the models.

## 🧪 Experiment 1

### Step 1 - Resource collection

This step collects the HuggingFace resources: the model, and the training dataset.

- `docker exec -it kafka python 0_data_collection.py`

A text sample generated from TinyLlama is provided. The initial prompt will always be the same: `K. ouvrit la porte.` (*K. opened the door.*), and the model used is always q8-quantized. 500 tokens are generated with a temperature of `0.8`, a repetition penalty of `1.1` and top-p of `0.9`. An English translation has been provided using ChatGPT, asking for close translation including errors and inconsistencies.

<details><summary>🇫🇷</summary>

```txt
— Je ne le crois pas, Maman.

— Tu veux qu'il te donne un coup d'œil ?

— Oh, non, pas de tout cela ! Je viens de voir une femme qui savait bien ce qu'elle avait à dire...

Il ne lâcha pas la main.

— Si tu avais vu des femmes, tu aurais vu des hommes aussi, mon fils !

— Quoi ?

— Si tu as vu des femmes, tu es vraiment de mauvaise humeur ; si tu as vu des hommes, tu es bien fort de meme, monsieur ! Tu ne crois pas que tu puises les femmes dans tes rêves ?

Le père se tourna vers lui :

— Maître, tu es vraiment un homme. C'est ce que je disais au premier jour de ta vie. Tu n'as pas pris garde à moi. Mais toi, tu n'aimes pas la fille. Elle vous fait peur, tu l'aurais préférée au docteur Culpepper, lequel vous connaissait depuis longtemps et dont il fallait se tuer pour s'en rendre compte.

Mademoiselle Chatillon reçut une telle réponse en sortant, elle se mit à la promener avec son petit cheval blanc, elle acheva de se rappeler cette scène de ses rêveries : c'était une sorte de contrainte, le docteur Culpepper avait dit qu'elle devrait être folle.

Après cette nuit de noces, Chatillon était de retour chez sa mère, puis il partit à Paris. Il eut pour compagnons de voyage M. et Mademoiselle Houdon et Monsieur de la Tour du Pin. Il y avait également un jeune officier, le maréchal de Campard. Au bout de quelques semaines, le comte de Montalivet fit venir son fils pour le conduire jusqu'à Paris.

Chatillon fut accueilli par une foule entière qui allait enfin admettre sa beauté et sa bonté, mais aussi la grande cruauté de son goût envers les femmes, car
```
</details>

<details><summary>🇬🇧</summary>
  
```txt
— I don't believe it, Mom.

— You want him to give you a glance?

— Oh, no, nothing of that! I just saw a woman who knew well what she had to say...

He didn’t let go of the hand.

— If you had seen women, you would have seen men too, my son!

— What?

— If you’ve seen women, you’re really in a bad mood; if you’ve seen men, you’re very strong the same, sir! Don’t you think you can draw women from your dreams?

The father turned toward him:

— Master, you really are a man. That’s what I was saying on the first day of your life. You didn’t take heed of me. But you, you don’t love the girl. She frightens you, you would have preferred her to Doctor Culpepper, who had known you for a long time and for whom one had to kill oneself to realize it.

Miss Chatillon received such an answer on going out, she began to walk it around with her little white horse, she finished remembering this scene from her reveries: it was a kind of constraint, Doctor Culpepper had said she would have to be crazy.

After this wedding night, Chatillon was back at his mother’s, then he left for Paris. His travel companions were Mr. and Miss Houdon and Monsieur de la Tour du Pin. There was also a young officer, Marshal Campard. After a few weeks, the Count of Montalivet sent for his son to lead him to Paris.

Chatillon was welcomed by an entire crowd that would at last admit his beauty and his goodness, but also the great cruelty of his taste toward women, for
```
</details>

### Step 2 - French literature aculturation

The goal of this step is to reorient the base model towards a generator of French literature. It consists in a full-weight training over 1M samples of 512 tokens from the Gallica collection. The model should forget its chatbot abilities, its multilinguism and its coding knowledge; to learn about *boudoir intrigues* and French classical literature content and form.

- `docker exec -it kafka python 1a_prepare_gallica_1M_512t.py`
- `docker exec -it kafka python 2a_train_tinyllama_fullweight_gallica_512t.py`

<div align="center">
<img width="500" alt="plot_train_fullweight" src="https://github.com/user-attachments/assets/3fa3ba32-f0e2-47e2-ad92-eb662e9128dd" />
</div>

Generation sample at this step (same parameterization than step 1):

<details> <summary>🇫🇷</summary>
  
```txt
— 5 — 
Qu'est-ce que cela? demanda l'homme d'Elsen. — Un homme. — Et quel est cet homme? — Un des plus riches du royaume, un noble. — Vous allez le voir ? — Oui, mais il n'y a pas longtemps qu'il est parti... Nous sommes au fond de la forêt. 
Oui, dit l'un d'eux, il y avait cinq jours que nous avions passé en courant sur cette route. 
— Votre maître est un brave et un honnête homme. — Vous savez ce qui vous sera arrivé. — Ce que j'aime à appeler une vertu, c'est de ne jamais se faire tromper. — Mais voici une fois qui m'est impossible. — Et qui donc? — Un noble de la maison d'Erthor. 
Cette question était énigmatique; il n'était point de la sagesse de M. 

— 6 — Lalande pour lui répondre : « Un noble de la maison d'Erthor. » Cela est bien étrange, car M. Lalande ne connaissait pas les gens de son pays, il était né dans une province, on ne l'avait pas vu, ni avec lequel on ne s'était pas fait connaître, et sa maison avait été détruite par les Anglais, par les Suisses, par ceux qui étaient venus après eux ; enfin, son père n'avait jamais vécu et son nom, lui aussi, n'était connu que par ses parents ; il était né en un village de Bourgogne, et comme il était tout en deuil, il l'avait donné à son père, son épouse était morte dans un grand accident, ils avaient eu quatre enfants, deux fils et deux filles, et il était tombé en laissant un seul enfant, une fille qui a trouvé une place chez un noble de la maison d'Erthor. — 

— 7 — Qui peut être celui de ces trois hommes? 
— Eh bien ! je le crois, sans doute, parce qu'une autre personne que celle qui a fait cette demande
```
</details>

<details><summary>🇬🇧</summary>
  
```txt
— 5 —
“What is that?” asked the man from Elsen.
— “A man.”
— “And what sort of man is that?”
— “One of the richest in the kingdom, a noble.”
— “You’re going to see him?”
— “Yes, but he hasn’t been gone long… We are at the bottom of the forest.”
“Yes,” said one of them, “it had been five days that we had run along this road.”
— “Your master is a brave and an honest man.”
— “You know what will have happened to you.”
— “What I like to call a virtue is never letting oneself be deceived.”
— “But here is one time that is impossible for me.”
— “And who then?”
— “A noble of the house of Erthor.”

This question was enigmatic; it was not in Mr.

— 6 — Lalande’s wisdom to answer him: “A noble of the house of Erthor.” That is very strange, for Mr. Lalande did not know the people of his own country, he had been born in a province, he had not been seen, nor had he made himself known to anyone, and his house had been destroyed by the English, by the Swiss, by those who had come after them; finally, his father had never lived and his name, too, was known only by his relatives; he had been born in a village of Burgundy, and as he was all in mourning, he had given it to his father, his wife had died in a great accident, they had had four children, two sons and two daughters, and he had fallen leaving only one child, a daughter who found a place with a noble of the house of Erthor. —

— 7 — “Who can that be, among these three men?”
— “Well! I believe it, no doubt, because another person than the one who made this request…”
```
</details>

### Step 3 - Strengthen the narrative arc

This step aims at teaching the model long (2048 tokens) and consistent narrative arcs, which is essential for a literature project. However, because the VRAM need increases quadratically with the context window, a QLoRA approach is undertaken from this step (and for the next one). LoRA adapters are trained over 100M samples of 2048, still from the Gallica collection.

- `docker exec -it kafka python 1b_prepare_gallica_100K_2048t.py`
- `docker exec -it kafka python 2b_train_gallica_2048t_QLoRA.py`

<div align="center">
<img width="500" alt="plot_train_qlora" src="https://github.com/user-attachments/assets/22440696-fd6e-494e-a348-ce4deefda5c6" />
</div>

Generation sample at this step (same parameterization than step 1):

<details> <summary>🇫🇷</summary>
  
```txt
BARONIS.
A la bonne heure, mon ami ! 
L'entendîmes à demi-voix; puis je me levai pour l'accompagner à sa chambre. 
MADAME HERMINE. 
Oh! si vous vouliez bien me suivre ; j'aurais un peu plus de confiance en votre amitié. 
(BARONIS entra dans la chambre de la comtesse.) 
BARONIS. 
Oui, monsieur, et j'attends un moment avec vous ? 

MOIRELLA. 
Pour vous raconter le récit de ces aventures qui m'ont été si vifement intéressantes dans son cours, mon cher monsieur. 
BARONIS. 
Je ne suis pas encore parvenu à les lire. 
MOIRELLA. 
Et il est assez difficile d'être réellement ému en voyant ce qu'on fait au public, sans avoir l'habitude du théâtre et de l'art dramatique. 
BARONIS. 
Il y a quelque temps que j'ai eu le plaisir de voir cette tragédie jouée devant mes camarades du corps militaire de Paris : c'est une production remarquable de M. Théodore de Banville, qui n'a rien d'étonnant au premier abord, car elle a tous les ressemblances avec la pièce que nous venons d'écouter. Il se trouve en effet que ce sont des personnages de la vie moderne, des figures des classes supérieures, des scènes du grand monde, d'une langue simple et concise, de l'art de s'exprimer comme l'on ne peut le faire dans d'autres circonstances, avec plus de finesse et de réalisme. 
MOIRELLA. 
Vous avez bien raison, monsieur; mais vous êtes trop fort pour être obligé de traiter les choses ainsi. 
BARONIS. 
Si je ne tins pas les yeux sur le spectacle, et que ce soit le jeu, je le croirais; et je me défie de vous appr
```
</details>

<details><summary>🇬🇧</summary>
  
```txt
BARONIS.
At last, my friend!
We heard him in a half-voice; then I got up to accompany him to his room.

MADAME HERMINE.
Oh! if you would be so kind as to follow me; I would have a bit more confidence in your friendship.

(BARONIS entered the countess’s room.)

BARONIS.
Yes, sir, and am I waiting a moment with you?

MOIRELLA.
To tell you the story of those adventures that were so vividly interesting to me as they happened, my dear sir.

BARONIS.
I have not yet managed to read them.

MOIRELLA.
And it is rather difficult to be truly moved when seeing what is shown to the public, without being used to the theatre and the dramatic art.

BARONIS.
Some time ago I had the pleasure of seeing that tragedy played before my comrades of the military corps of Paris: it is a remarkable work by Mr. Théodore de Banville, which is not surprising at first sight, for it has all the resemblances with the play we have just listened to. Indeed, it turns out they are characters of modern life, figures of the upper classes, scenes of high society, with a simple and concise language, with the art of expressing oneself as one cannot do in other circumstances, with more subtlety and realism.

MOIRELLA.
You are quite right, sir; but you are too strong to be forced to handle things in that way.

BARONIS.
If I did not keep my eyes on the show, and if it were just the acting, I would believe it; and I mistrust telling you
```
</details>

### Step 4 - Stylistic imprinting on Kafka

Finally, the French litterature model and more specifically its previously pre-trained LoRA are fine-tuned on the target book: the French translation of The Castle by Kafka. This step takes as input 2048 token long sequences of the book, with a stride of 512, yielding 4 shuffled "pseudo-epochs" (each token of the book is seen 4 times).

- `docker exec -it kafka python 1c_prepare_kafka_2048t.py`
- `docker exec -it kafka python 2c_train_kakfa_QLoRA.py`

<div align="center">
<img width="500" alt="plot_train_kafka" src="https://github.com/user-attachments/assets/4338ce8e-8a25-4e49-9a3e-ab97f205c07e" />
</div>

Generation sample at this step (same parameterization than step 1):

<details><summary>🇫🇷</summary>

```txt
25 
C'était la maison de mon oncle, qui était mort il y a une année environ. Je me trouvais le premier auprès de lui dans sa chambre, quand j'aperçus quelqu'un que je ne m'attendais pas, et qui l'attira par la main. Il s'approcha d'eux tous deux et ils mirent les bras autour des corps. Le lendemain matin, je vis mon oncle, comme j'en avais vu autrefois; mais c'étaient ses jambes qui avaient été coupées en deux, et ses cheveux noirs sur son front, ainsi qu'il s'appelle, avec ces cheveux blancs qui tombent aux pieds, un peu grisés. Ils étaient tous deux à cheval, sur un mouton blanc, et se regardaient attentivement. Un ruisseau coulait devant nous; tout était rouge, et un grand nuage de poussière descendait du ciel. Cela n'allait pas assez loin pour qu'il eût fallu une bouteille de vin pour relever leur tempérament. 

— 26 — 
Les derniers jours de décembre, les vieilles gens sont dites fort gaielles; car il est plus doux de vivre avec le bonheur d'autrui qu'avec celui de soi-même. Nous avions à nous remettre ce jour-là, et nous nous étions réfugiés chez un de mes amis, que nous appelions de nos propres yeux monsieur le curé. Ce fut le dernier soir de ma présence ici, et je suis venu à Paris pour faire mon voyage. Quelque temps après, mon oncle et notre ami étant revenus à leur poste, on fit quelques choses en parlant ensemble. Après avoir pris quelque chose de boisson, ils allèrent à l'église. A peine avait-ils quitté cette ville et ces terres, qu'ils eurent la fièvre. J'avais eu le malheur de voir mon oncle malade, mais j'étais si faible, qu'au moment où il était très malade, je croyais
```
</details>

<details><summary>🇬🇧</summary>
  
```txt
25
It was my uncle’s house, who had died about a year ago. I was the first beside him in his room, when I noticed someone I was not expecting, and who pulled him by the hand. He came closer to the two of them and they put their arms around the bodies. The next morning, I saw my uncle, as I had seen him before; but it was his legs that had been cut in two, and his black hair on his forehead, as he is called, with those white hairs that fall to the feet, a little greyed. They were both on horseback, on a white sheep, and were looking at each other attentively. A stream was flowing before us; everything was red, and a great cloud of dust was coming down from the sky. It didn’t go far enough for them to need a bottle of wine to lift their spirits.

— 26 —
In the last days of December, old people are said to be very merry; for it is sweeter to live with someone else’s happiness than with one’s own. We had to pull ourselves together that day, and we had taken refuge at one of my friends’, whom we called with our own eyes Mister Curé. That was the last evening of my presence here, and I came to Paris to make my journey. Some time after, my uncle and our friend having returned to their post, they did some things while talking together. After having taken something to drink, they went to the church. Barely had they left this town and these lands, when they caught the fever. I had had the misfortune of seeing my uncle ill, but I was so weak that at the moment when he was very ill, I thought…
```
</details>

### 📊 Conclusions

The French literary aculturation (step 1) was very effective. The model completely switched its register, using a highly elevated language register and a specific vocabulary of the French XIXth literature. The number of training samples could have been shortened, as the loss stagnated for more than half the training.

Step 2 and 3 were disappointing. For step 2, either the undertaken QLoRA approach was not effective with this configuration, either the longer attention was not degraded at step 1, in any case no sustantial gain was observed. Step 3 was clearly not daring enough:  loss variations were erratic, and the model failed to generate Castle-related content.

## 🧪 Experiment 2 - Full-weight fine-tuning after aculturation

Given the failure of the QLoRA approach, a full-weight training was directly attempted on the target book. The model from experiment 1 - Step 1, *i.e.* the French literary aculturated model, was taken as the base model for this second experiment.

The training on The Castle was more aggressive: it was split into 512 tokens chunks, shuffled, and used as training material along 10 epochs. The goal of this step is to play with the edge of overfitting, in other words, to deeply imprint the target book without yielding a perfect recitation.

- `docker exec -it kafka python 1d_prepare_kafka_512t.py`
- `docker exec -it kafka python 2d_train_kafka_fullweight_512t.py`

<div align="center">
<img width="500" alt="plot_train_kafka" src="https://github.com/user-attachments/assets/8303429f-cd22-4b83-afc0-087fe3c84990" />
</div>

<details><summary>🇫🇷</summary>

```txt
– Monsieur l’Arpenteur ? demanda-t-il en se penchant à petits coups dans le lit, sans même toucher K., mais sans pouvoir s’empêcher d’en tirailler les cheveux. Dans quel monde ? Quand je vous vois encore si malade, quand je vous retrouve au milieu de cette nuit froide et vide, quand j’entends entendre mes messieurs, qui sont tous là-bas, discuter de moi comme ils eussent discuté de vous – ils n’ont pas oublié votre place, mais ils ont fait un gros geste pour ne pas te la lui remettre ! –, quand je vous vois ainsi, même si tu n’étais plus capable de rien faire, il est donc impossible que tu me rendes compte qu’un tel châtiment ait été fait pour toi. Il ne suffit pas de n’être pas le premier des gens, il faut aussi d’être digne d’y être. Je vous ai pris pour un valet de chambre, je n’ai jamais osé croire qu’on m’eût cru capable d’un pareil travail ; c’est ce que je disais autrefois même aux gens qui voulaient m’embaucher, mais maintenant je sais que je vous ai pris pour un valet de chambre, vous avez seulement manqué de qualités nécessaires à cet emploi. Vous n’avez pas su rester longtemps dans une pièce où on commençait par vous mettre à nu, vous n’aviez pas la seule force de le faire. C’est au fond du moins ma faute, car, quand j’ai appris vos défauts, je vous ai mis dans la cour du château et j’ai fait tout mon possible pour ne pas vous voir. Mais cela ne suffit pas, et il ne peut être question que d’une chose plus grande : je suis méchante avec vous, je vous traite mal, je ne puis me trouver à ta place et n’ai aucun désir de devenir celle-là. Tu es donc reparti ?
– Oui, dit K. en regardant devant lui.
– Étrange jeune homme, dit Olga, je l’
```
</details>

<details><summary>🇬🇧</summary>

```txt
– “Land-Surveyor, sir?” he asked, leaning in little dips over the bed, without even touching K., yet unable to stop tugging at his hair. “In what world? When I still see you so sick, when I find you again in the middle of this cold and empty night, when I hear my gentlemen, who are all down there, discussing me as they would have discussed you—they haven’t forgotten your place, but they made a big gesture not to give it back to him!—when I see you like this, even if you were no longer capable of doing anything, how is it possible you cannot explain to me that such a punishment was done for you. It isn’t enough not to be the first of people, you must also be worthy of being among them. I took you for a valet, I never dared believe one could have thought me capable of such a piece of work; that is what I used to tell even to people who wanted to hire me, but now I know I took you for a valet, you merely lacked the qualities necessary for that position. You didn’t know how to stay long in a room where they began by stripping you naked, you didn’t have the slightest strength to do it. It is, at bottom, my fault at least, for when I learned your shortcomings, I put you in the castle’s courtyard and did everything in my power not to see you. But that isn’t enough, and there can be talk only of something greater: I am wicked with you, I treat you badly, I cannot put myself in your place and have no desire to become that. So you have gone away again?”
– “Yes,” said K., staring ahead of him.
– “Strange young man,” said Olga, “I…”
```
</details>

### 📊 Conclusions

This second experiment was clearly more convincing: the characters and vocabulary are recurrently taken from The Castle, and the phrasing is much more Kafkaesque. We stopped at 10 epoch but the loss kept decreasing; there is a gradient of imprinting that may be played with, from zero to pure memorization of the book. 10 epochs likely crosses into the audacious zone, the narrative thread is a bit disjointed and the model sometimes goes off the rails, for instance starting to declaim theatre.

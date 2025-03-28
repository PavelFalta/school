# SWI ZAPISKY
### ukonceni
- seminarni prace
### zkouska
- ustni obhajoba


### lit 
- sommerville ian - software engineering (prvni sekce, asi precist tbh)

# 1
### systemove inzenyrstvi
- implementace/testovani/udrzba

### co je to software?
- podle sommervila: kod, dokumentace, lide
- programovani prekvapive mala podmnozina
- softwarove inzenyrstvi se zabiva vsemi aspekty vyvoje

### model procesu vyvoje softwaru
- model, protoze se jedna o "zjednodusenou realitu"

#### kroky:
    specifikace
    implementace
    validace
    evoluce

### SCRUM

- role
- artefakt
- pravidla


### Typy modelu
- agilni vyvoj
```
https://agilemanifesto.org
```
#### vodopoadovy model
- take znam jako SDLC
- naprd :p (moc expensive delat zmeny hluboko ve vyvoji)

#### V-model
- lepsi ale starej furt
- velka testovaci vetev
- validace paralelne s vyvojem
- zavedl pojmy

#### inkrementalni vs iterativni

#### boehm spiral

#### prototype model


#### rapid application development

#### RUP

## agilni metodiky
#### extreme programming
#### scrum
#### kanban

# 2,

## MPS
- S -> I -> V -> E
### inzenyrstvi pozadavku
- klient nevi co chce
#### pozadavek
- atom, objekt se kterym pracujeme. neco, co klient chce od systemu ktery vytvarime
1. funkcni
    - co chci
    - chci aby mi system delal tohle a tohle
2. mimofunkcni
    - vlastnosti
    - efektivity

#### hodne velke meritko
1. studium proveditelnosti
2. elicitace a analyza pozadavku
3. specifikace pozadavku (formalni zapis)

#### 2 dokumentace
- jedna pro programatory a uzivatele

### mimo-funkcni pozadavky
- MFP se rozdeluji na
1. produktove (kvalita, spolehlivost, bezpecnost, efektivita (CPU/RAM))
2. organizacni (omezeni, operacni systemy/zarizeni, responsibilita, vyvojarske, 2FA...)
3. externi (zakon, regulace, etika)


### co je nejhorsi
- vsechno tohle predchozi chceme, aby bylo kvantifikovatelne (dukazy)
- abychom overili, ze jsme splnili mimo-funkcni pozadavky


## Berankovo template struktury pozadavku
1. preface
2. uvod
3. glosar
4. funkcni pozadavky
5. diagramy 
6. mimo funkcni pozadavky
7. evoluce
8. prilohy
9. index

### modelovani pozadavku

1. NLP
2. strukturovany zapis (user stories)
3. DJS (i*), ve vyvoji zatim
4. diagramy
5. matematika

# 3,
- systemove modelovani

## k cemu?
1. vyjednavani R
2. dokonceni sw arch
3. voditko pro programatory
4. voditko pro testing
5. voditko pro evoluci
6. voditko pro rizeni

## typy
1. externi
    - kontextovy
2. interakcni
    - use-case, sekvencni
3. strukturalni
    - class diagram
4. behavioralni
    - aktivity, stavy

### nejdulezitejsi podle berankovo clanku/pruzkumu
1. **use case diagram** pro vyjadreni interakce uzivatele se systemem
2. **sekvencni** pro vyjadreni chovani vevnitr
3. **class diagram** pro vyjadreni struktury

## teorie modelovani (podle beranka)
1. **DDM** = data driven modelling (1970...)
    - data-flow diagram
2. **EDM** = event-driven modelling
    - stavove diagramy
3. **MDE** = modely rizene inzenyrstvi
    - dulezite je model, az druhotne kod
    - vyvojar by tedy mel produkovat graficke modely
    - vzniklo MDA od OMG
    - 3 casti MDA:
        1. CIM
        2. PIM
        3. PSM
    - revival pomoci LLM?
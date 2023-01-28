
def get_pollen_family(pollen_type):

    if pollen_type in ['Acanthus dioscoridis', 'Acanthus sp']:
        return 'Acanthaceae'
    elif pollen_type in ['Sambucus ebulus', 'Sambucus nigra', 'Viburnum lantana','Viburnum lanata']:
        return 'Adoxaceae'
    elif pollen_type in ['Chenopodium foliosum','Amaranthaceae sp']:
        return 'Amaranthaceae'
    elif pollen_type in ['Allium rotundum', 'Allium ampeloprasum', 'Allium sp']:
        return 'Amaryllidaceae'
    elif pollen_type in ['Cotinus coggygria', 'Rhus coriaria', 'Rhus sp']:
        return 'Anacardiaceae'
    elif pollen_type in ['Eryngium campestre', 'Ferula orientalis', 'Malabaila lasiocarpa', 'Lecokia cretica', 'Pimpinella sp','Apiaceae sp']:
        return 'Apiaceae'
    elif pollen_type in ['Leopoldia tenuiflora','Ornithogalum narbonense','Muscari tenuifolium']:
        return 'Asparagaceae'
    elif pollen_type in ['Gundelia sp','Circium arvense','Centaurea urvellei','Cardus','Matricaria chamomila','Scorzonera latifolia','Asteraceae sp','Artemisia absinthium','Achillea arabica', 'Achilla arabica', 'Achillea millefolium', 'Achillea vermicularis','Anthemis cretica','Arctium minus','Arctium minus','Bellis perennis',
    'Carduus nutans','Centaurea bingoelensis','Centaurea iberica','Centaurea kurdica','Centaurea saligna','Centaurea solstitialis','Centaurea spectabilis',
    'Centaurea urvillei','Centaurea virgata','Chondrilla brevirostris','Chondrilla juncea','Cichorium intybus','Cirsium arvense','Cirsium yildizianum',
    'Cota altissima','Crepis sancta','Cyanus triumfettii','Echinops pungens','Gundelia tournefortii','Helichrysum arenarium','Helichrysum plicatum',
    'Iranecio eriospermus','Senecio eriospermus','Matricaria chamomilla','Onopordum acanthium','Onopordum acanthium','Tanacetum balsamita','Tanacetum zahlbruckneri',
    'Taraxacum campylodes','Taraxacum officinale','Tussilago farfara','Xeranthemum annuum','Xeranthemum longipapposum','Artemisia sp','Carduus sp','Centaurea sp',
    'Cichorium sp','Cirsium sp','Cirsium yildizianum','Echinops sp','Echinops sp','Helianthus sp','Helichrysum sp','Onopordum sp','Ptilostemon sp','Xanthium sp','Xeranthemum sp','Xeranthemum longipopposum','Xeranthemum annum']:
        return 'Asteraceae'
    elif pollen_type in ['Alkanna orientalis','Anchusa azurea','Anchusa leptophylla','Cerinthe minor','Echium italicum','Myosotis alpestris','Myosotis laxa','Myosotis stricta','Myosotis sylvatica','cyanea','Phyllocara aucheri','Anchusa sp','Echium sp']:
        return 'Boraginaceae'
    elif pollen_type in ['Capsella bursa pastoris','Aethionema grandiflorum', 'Capsella bursa-pastoris', 'Isatis glauca','Lepidium draba','Raphanus raphanistrum']:
        return 'Brassicaceae'
    elif pollen_type in ['Campanula glomerata','Campanula involucrata', 'Campanula propinqua','Campanula stricta', 'stricta', 'Campanula sp']:
        return 'Campanulaceae'
    elif pollen_type in ['Centranthus longifolius','Centranthus longiflorus', 'Morina persica', 'Scabiosa columbaria', 'Scabiosa rotata', 'Cephalaria sp', 'Scabiosa sp', 'Valeriana sp']:
        return 'Caprifoliaceae'
    elif pollen_type in ['Cerastium armeniacum', 'Saponaria prostrata', 'Saponaria viscosa', 'Saponaria viscosa', 'Silene spergulifolia','Dianthus sp','Silene sp','Silene compacta']:
        return 'Caryophyllaceae'
    elif pollen_type in ['Chenopodium sp']:
        return 'Chenopodiaceae'
    elif pollen_type in ['Convolvulus arvensis', 'Convolvulus galaticus', 'Convolvulus lineatus', 'Convolvulus sp']:
        return 'Convolvulaceae'
    elif pollen_type in ['Cornus sanguinea']:
        return 'Cornaceae'
    elif pollen_type in ['Phedimus obtusifolius']:
        return 'Crassulaceae'
    elif pollen_type in ['Elaeagnus angustifolia']:
        return 'Elaeagnaceae'
    elif pollen_type in ['Euphorbia esula','tommasiniana', 'Euphorbia macrocarpa', 'Euphorbia sp']:
        return 'Euphorbiaceae'
    elif pollen_type in ['Glycyrrhiza glabra','Astragalus topolanense','Astragalus sp','Astragalus pinetorium','Astragalus lagopoides','Securigera varia','Astracantha gummifera','Trifolium campestre-yeniden','Astragalus gummifer','Astracantha kurdica','Astragalus kurdicus','Astracantha muschiana','Astragalus muschianus','Astragalus aduncus','Astragalus bingollensis','Astragalus bustillosii','Astragalus brachycalyx','brachycalyx','Astragalus caspicus','Astrgalus lagopoides','Astrgalus lagopoides Lam','Astragalus onobrychis','Astragalus oocephalus','Astragalus pinetorum','Astragalus saganlugensis','Astragalus topalanense','Colutea cilicica','Genista aucheri','Genista aucheri','Lathyrus brachypterus','Lathyrus satdaghensis','Lotus corniculatus','Lotus gebelia','Hedysarum varium','Medicago sativa','Melilotus albus','Melilotus officinalis','Onobrychis viciifolia','Ononis spinosa','Robinia pseudoacacia','Robinia pseudoacacia','Trifolium campestre','Trifolium diffusum','Trifolium nigrescens','Trifolium pauciflorum','Trifolium pratense','Trifolium resupinatum','Vicia cracca','cracca','Astragalus gummifer','Astragalus longifolius','Astragalus topalanense','Astragalus spp','Coronilla sp','Hedysarum sp','Lathyrus sp','Lotus sp','Melilotus sp','Trifolium spp','Vicia sp']:
        return 'Fabaceae'
    elif pollen_type in ['Quercus petraea', 'pinnatiloba']:
        return 'Fagaceae'
    elif pollen_type in ['Geranium tuberosum', 'Geranium sp']:
        return 'Geraniaceae'
    elif pollen_type in ['Hypericum sp','Hypericum lydium', 'Hypericum perforatum','Hypericum scabrum','Hypericum spp']:
        return 'Hypericaceae'
    elif pollen_type in ['Ixiolirion tataricum']:
        return 'Ixioliriaceae'
    elif pollen_type in ['Marrubium astracanium','Salvia palestina','Ajuga chamaepitys','chia','Lamium album','Lamium garganicum','Lamium macrodon','Marrubium astracanicum','Marrubium vulgare','Mentha longifolia','longifolia','Mentha spicata','Nepeta baytopii','Nepeta cataria','Nepeta nuda','Nepeta trachonitica','Origanum acutidens','Origanum vulgare','gracile','Phlomis armeniaca','Phlomis herba-venti','pungens','Phlomis pungens','Phlomis kurdica','Salvia frigida','Salvia limbata','Salvia macrochlamys','Salvia multicaulis','Salvia palaestina','Salvia sclarea','Salvia staminea','Salvia trichoclada','Salvia verticillata','Salvia virgata','Satureja hortensis','Stachys annua','Stachys lavandulifolia','Teucrium chamaedrys','Teucrium orientale','Teucrium orientale','Teucrium polium','Thymus kotschyanus','Thymus pubescens','Lamium macrodon','Lamium sp','Origanum sp','Phlomis sp','Teucrium sp','Thymus spp']:
        return 'Lamiaceae'
    elif pollen_type in ['Linum mucronatum','armenum']:
        return 'Linaceae'
    elif pollen_type in ['Lythrum salicaria']:
        return 'Lythraceae'
    elif pollen_type in ['Malvaceae','Alcea apterocarpa','Alcea remotiflora','Malva neglecta','Alcea sp','Malva sp','Tilia sp']:
        return 'Malvaceae'
    elif pollen_type in ['Morus alba']:
        return 'Moraceae'
    elif pollen_type in ['Epilobium parviflorum']:
        return 'Onagraceae'
    elif pollen_type in ['Fumaria parviflora','Fumaria schleicheri','microcarpa','Papaver dubium','Papaver orientale']:
        return 'Papaveraceae'
    elif pollen_type in ['Anarrhinum orientale','Anarhinum orientale','Globularia trichosantha','Lagotis stolonifera','Linaria pyramidata','Plantago lanceolata','Plantago major','Plantago media','Linaria sp']:
        return 'Plantaginaceae'
    elif pollen_type in ['Acantholimon acerosum','Acantholimon armenum','Acantholimon calvertii','Acantholimon sp']:
        return 'Plumbaginaceae'
    elif pollen_type in ['Poaceae','Zea mays']:
        return 'Poaceae'
    elif pollen_type in ['Polygonum cognatum','Rheum ribes','Rumex acetosella','Rumex scutatus','Rumex sp']:
        return 'Polygonaceae'
    elif pollen_type in ['Lysimachia punctata', 'Lysimacha vulgaris']:
        return 'Primulaceae'
    elif pollen_type in ['Portulaca sp']:
        return 'Portulacaceae'
    elif pollen_type in ['Ranunculus kotchii','Ficaria fascicularis','Ranunculus kochii','Ranunculus heterorrhizus','Ranunculus kotschyi','Ranunculus sp']:
        return 'Ranunculaceae'
    elif pollen_type in ['Paliurus spina-christi']:
        return 'Rhamnaceae'
    elif pollen_type in ['Crateagus orientalis','Crateagus monogyna','Potentilla inclinata','Rosaceae','Sanguisorba minör','Sanguisorba min”r','Agrimonia repens','Cotoneaster nummularius','Crataegus orientalis','Crataegus monogyna','Filipendula ulmaria','Malus sylvestris','Potentilla anatolica','Potentilla argentea','Prunus divaricata','ursina','Pyrus elaeagnifolia','Rosa canina','Rosa foetida','Rubus caesius','Rubus sanctus','Sanguisorba minor','lasiocarpa','Sorbus torminalis','Filipendula sp','Potentilla sp','Rosa canina']:
        return 'Rosaceae'
    elif pollen_type in ['Galium consanguineum','Galium verum','Galium sp']:
        return 'Rubiaceae'
    elif pollen_type in ['Citrus sp']:
        return 'Rutaceae'
    elif pollen_type in ['Salix alba','Salix caprea','Salix sp']:
        return 'Salicaceae'
    elif pollen_type in ['Verbascum armenum','Verbascum diversifolium','Verbascum gimgimense','Verbascum lasianthum','Verbascum sinuatum','Verbascum spp','Verbascum sinatum']:
        return 'Scrophulariaceae'
    elif pollen_type in ['Tamarix smyrnensis','Tamarix tetrandra']:
        return 'Tamaricaceae'
    elif pollen_type in ['Eremurus spectabilis','Eremurus sp']:
        return 'Xanthorrhoeaceae'
    elif pollen_type in ['Tribulus terrestris']:
        return 'Zygophyllaceae'
    else:
        return 'notfound'


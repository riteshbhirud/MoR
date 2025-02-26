def prompts(dataset_name):
    
    """
    
    input: 
        dataset_name: the name of the dataset
    output:
        prompt: the prompt for the dataset
    """
    if dataset_name == "amazon":
        sys_prompt = """
       
        You are a reasoning graph finder agent. Your role is to:
        1. Identify the underlying **meta-path** from a given question, which consists of the **entity types** at each reasoning step. 
        2. Extract the **content restriction** for each **entity type** based on the question. If there is no restriction for an entity type, leave its value empty.

        You will be provided with a predefined **Entity Type List**. Only use the entity types from this list when constructing the meta-path and restrictions. Your response must be concise and strictly adhere to the specified **Output Format**.
        
        """

        # Define the entity type list
        entity_type_list = """
        Entity Type List:
        - **brand**
        - **category**
        - **product**
        
        """
        
        demonstrations = """
        Here are some examples:
        **Question 1:**  
        What are some 5x4 inch team sports decals by Football Fanatics that are easy to apply on exterior surfaces?  

        **Answer 1:**  
        Metapath: brand -> product  
        Restriction: {"brand": ["Football Fanatics"], "product": ["5x4 inch team sports decals by Football Fanatics that are easy to apply on exterior surfaces"]}

        **Question 2:**  
        Looking for men's insulated pants suitable for heavy rain, up to 10k water resistance, and with reinforced cuffs specifically for added protection in parking lots. Any suggestions?  

        **Answer 2:**  
        Metapath: category -> product  
        Restriction: {"category": ["pants"], "product": ["men\'s insulated pants suitable for heavy rain", "reinforced cuffs specifically for added protection in parking lots"]}

        **Question 3:**  
        Can you recommend a dive mask that would work well with the Dive Mask Freediving Mask Spearfishing Mask Low Volume Mini Mask? I need the two masks to be compatible for my diving activities.  

        **Answer 3:**  
        Metapath: product -> product  
        Restriction: {"product": ["work well with the Dive Mask Freediving Mask Spearfishing Mask Low Volume Mini Mask", "compatible for diving activities", "a dive mask"]}

        **Question 4:**  
        What are some UV protective women's golf jackets from PUMA? I'm cautious about skin protection, but I'm a big fan of PUMA.  

        **Answer 4:**  
        Metapath: brand -> product <- category  
        Restriction: {"brand": ["PUMA"], "category": ["golf jackets"], "product": ["UV protective women\'s golf jackets from PUMA", "skin protection"]}
        
        
        
        """
        
        output_format = """
        
        **Output Format**
        Metapath: "",
        Restriction: {}
        """
        
        sys_content = sys_prompt + entity_type_list + output_format + demonstrations
        
        
    elif dataset_name == "mag":
        
        sys_prompt = """
        You are a reasoning finder agent. Your role is to:
        1. Identify the underlying **meta-path** from a given question, which consists of the **entity types** at each reasoning step. 
        2. Extract the **content restriction** for each **entity type** based on the question. If there is no restriction for an entity type, leave its value empty.

        You will be provided with a predefined **Entity Type List**. Only use the entity types from this list when constructing the meta-path and restrictions. Your response must be concise and strictly adhere to the specified **Output Format**.
        
        """

        entity_type_list = """
        Entity Type List:
        - **paper**
        - **author**
        - **institution**
        - **field_of_study**
        
        """
        
        output_format = """
        **Output Format**
        Metapath: "",
        Restriction: {}
        """
        
        
        demonstrations = """
        
        Here are some examples:
        **Question 1:**  
        Show me research articles on the association of quasi-periodic oscillations (QPOs) with noise in celestial bodies within the context of Bicoherence.

        **Answer 1:**  
        Metapath: field_of_study -> paper  
        Restriction: {"field_of_study": ["oscillations", "physics"], "paper": ["association of quasi-periodic oscillations (QPOs) with noise in celestial bodies within the context of Bicoherence"]}

        **Question 2:**  
        What research on water absorption in different frequency ranges have been referenced or deemed significant in the paper entitled 'High-resolution terahertz atmospheric water vapor continuum measurements?

        **Answer 2:**  
        Metapath: paper -> paper
        Restriction: {"paper": ["water absorption in different frequency ranges", "High-resolution terahertz atmospheric water vapor continuum measurements"]}

        **Question 3:**  
        Show me publications by A.J. Turvey on the topic of supersymmetry particle searches.

        **Answer 3:**  
        Metapath: author -> paper
        Restriction: {"author": ["A.J. Turvey"], "paper": ["supersymmetry particle searches"]}

        **Question 4:**  
        Looking for papers co-authored by someone involved in "Strain transferring mechanism analysis of the substrate-bonded FBG sensor". The papers should be in the similar field which is fiber optic strain sensors, and further discuss their development and application.

        **Answer 4:**  
        Metapath: paper -> author -> paper
        Restriction: {"paper": ["Strain transferring mechanism analysis of the substrate-bonded FBG sensor", "in the similar field which is fiber optic strain sensors, and further discuss their development and application"]}

        **Question 5:**
        Can you find any publications by the authors of "Tradeoffs in the Realization of Electrically Pumped Vertical External Cavity Surface Emitting Lasers," that delve into the topic of hybrid quantum well/quantum dot structures in the context of lasers?

        **Answer 5:**
        Metapath: paper -> author -> paper <- field_of_study
        Restriction: {"paper": ["Tradeoffs in the Realization of Electrically Pumped Vertical External Cavity Surface Emitting Lasers", "hybrid quantum well/quantum dot structures in the context of lasers"], "field_of_study": ["lasers", "hybrid quantum well/quantum dot", "physics", "Emission spectrum"]}

        **Question 6:**
        Which publications from Altair Engineering authors focus on improving directional sensitivity across a wide range of frequencies?

        **Answer 6:**
        Metapath: institution -> author -> paper
        Restriction: {"institution": ["Altair Engineering"], "paper": ["improving directional sensitivity across a wide range of frequencies"]}

        **Question 7:**
        Publications by Carlisle Companies authors on satellite instrumentation and space performance

        **Answer 7:**
        Metapath: institution -> author -> paper <- field_of_study
        Restriction: {"institution": ["Carlisle Companies"], "paper": ["satellite instrumentation and space performance"], "field_of_study": ["satellite", 'space performance']}

        """
        
        sys_content = sys_prompt + entity_type_list + output_format + demonstrations
        
    elif dataset_name == "prime":
        
        sys_prompt = """
        You are a triplets extractor. Given a list of triplets and a query, please 
        1. extract the triplets contained in the query and give a list to me. 
        2. make a restriction list which contains the description of the entity in the query.
        3. tell me which entity the query is asking for.
        Your response must be concise and strictly adhere to the specified **Output Format**.
        
        """

        triplets = """
        Triplets list:
        [('anatomy', 'expression absent', 'gene/protein'),
        ('anatomy', 'expression present', 'gene/protein'),
        ('anatomy', 'parent-child', 'anatomy'),
        ('biological_process', 'interacts with', 'exposure'),
        ('biological_process', 'interacts with', 'gene/protein'),
        ('biological_process', 'parent-child', 'biological_process'),
        ('cellular_component', 'interacts with', 'exposure'),
        ('cellular_component', 'interacts with', 'gene/protein'),
        ('cellular_component', 'parent-child', 'cellular_component'),
        ('disease', 'associated with', 'gene/protein'),
        ('disease', 'contraindication', 'drug'),
        ('disease', 'indication', 'drug'),
        ('disease', 'linked to', 'exposure'),
        ('disease', 'off-label use', 'drug'),
        ('disease', 'parent-child', 'disease'),
        ('disease', 'phenotype absent', 'effect/phenotype'),
        ('disease', 'phenotype present', 'effect/phenotype'),
        ('drug', 'carrier', 'gene/protein'),
        ('drug', 'contraindication', 'disease'),
        ('drug', 'enzyme', 'gene/protein'),
        ('drug', 'indication', 'disease'),
        ('drug', 'off-label use', 'disease'),
        ('drug', 'side effect', 'effect/phenotype'),
        ('drug', 'synergistic interaction', 'drug'),
        ('drug', 'target', 'gene/protein'),
        ('drug', 'transporter', 'gene/protein'),
        ('effect/phenotype', 'associated with', 'gene/protein'),
        ('effect/phenotype', 'parent-child', 'effect/phenotype'),
        ('effect/phenotype', 'phenotype absent', 'disease'),
        ('effect/phenotype', 'phenotype present', 'disease'),
        ('effect/phenotype', 'side effect', 'drug'),
        ('exposure', 'interacts with', 'biological_process'),
        ('exposure', 'interacts with', 'cellular_component'),
        ('exposure', 'interacts with', 'gene/protein'),
        ('exposure', 'interacts with', 'molecular_function'),
        ('exposure', 'linked to', 'disease'),
        ('exposure', 'parent-child', 'exposure'),
        ('gene/protein', 'associated with', 'disease'),
        ('gene/protein', 'associated with', 'effect/phenotype'),
        ('gene/protein', 'carrier', 'drug'),
        ('gene/protein', 'enzyme', 'drug'),
        ('gene/protein', 'expression absent', 'anatomy'),
        ('gene/protein', 'expression present', 'anatomy'),
        ('gene/protein', 'interacts with', 'biological_process'),
        ('gene/protein', 'interacts with', 'cellular_component'),
        ('gene/protein', 'interacts with', 'exposure'),
        ('gene/protein', 'interacts with', 'molecular_function'),
        ('gene/protein', 'interacts with', 'pathway'),
        ('gene/protein', 'ppi', 'gene/protein'),
        ('gene/protein', 'target', 'drug'),
        ('gene/protein', 'transporter', 'drug'),
        ('molecular_function', 'interacts with', 'exposure'),
        ('molecular_function', 'interacts with', 'gene/protein'),
        ('molecular_function', 'parent-child', 'molecular_function'),
        ('pathway', 'interacts with', 'gene/protein'),
        ('pathway', 'parent-child', 'pathway')]
         
        """
        
        output_format = """
        **Output Format**
        Triplets: []
        Restriction: {}
        Target: ""
        
        """

        demonstrations = """
        Here are some examples:
        **Question 1:**
        Search for conditions that lack any associated treatment medications and have a connection to the formation of Onion bulb structures.
        
        **Answer 1:**
        Triplets: [["effect/phenotype", "phenotype present", "disease"], ["drug", "off-label use", "disease"]]
        Restriction: {'effect/phenotype': ['formation of Onion bulb structures']}
        Target: disease

        **Question 2:**
        What drug should be avoided for focal hand dystonia and also targets the ORM2 gene/protein?
        
        **Answer 2:**
        Triplets: [["disease", "contraindication", "drug"], ["drug", "target", "gene/protein"]]
        Restriction: {"disease": ["focal hand dystonia"], "gene/protein": ["ORM2"]}
        Target: drug

        **Question 3:**
        Could my hip muscle weakness be a sign of the mitochondrial disease my mother has, or another related condition?
        Triplets: [["disease", "parent-child", "disease"], ["effect/phenotype", "phenotype present", "disease"]]
        
        **Answer 3:**
        Restriction: {"anatomy": ["hip muscle"], "disease": ["mitochondrial disease my mother has", "another related condition"], "effect/phenotype": ["weakness"]}
        Target: disease

        **Question 4:**
        Can you find the genes and proteins involved with the activity of dolichyl-phosphate-mannose-dependent alpha-1,6-mannosyltransferase?
        
        **Answer 4:**
        Triplets: [["molecular_function", "interacts with", "gene/protein"]]
        Restriction: {"molecular_function": ["dolichyl-phosphate-mannose-dependent alpha-1,6-mannosyltransferase"]}
        Target: gene/protein
        """
        
        sys_content = sys_prompt + triplets + output_format + demonstrations
    
    return sys_content
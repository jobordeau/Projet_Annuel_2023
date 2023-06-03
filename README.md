# Projet_Annuel_2023
Pour le PMC il faut normalement mettre les vecteurs d'entrées dans  std::vector<std::vector<double>> all_samples_inputs = {} 
Et les sorties attendues il faut les mettre dans std::vector<std::vector<double>> all_samples_expected_outputs = {}
Il faut également modifié les arguments de la fonction train en fonction de la tache à faire i.e Classifaction, Régression
  e.g 
   bool is_classification = true;
   int iteration_count = 1000;
   double alpha = 0.2;
  
 Pour le model linéaire:
   il faut normalement mettre les vecteurs d'entrées dans std::vector<std::vector<double>> classificationData = {}
   Et les sorties attendues il faut les mettre dans std::vector<std::vector<double>> classificationTargets = {}
   A modifier les valeurs du learning_rate et le le bias et la tache à faire: Classification ou régression(true ou false)
   LinearModel classificationModel(classificationData.size(), 0.3, 0.5, true);
  
  
   
  
  

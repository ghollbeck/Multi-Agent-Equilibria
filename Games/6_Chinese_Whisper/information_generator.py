import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class InformationSeed:
    """Container for seed information with metadata."""
    content: str
    information_type: str
    complexity_level: str
    expected_key_elements: List[str]
    target_length: int

class InformationGenerator:
    """
    Generates seed information of varying complexity and types for the Chinese Whisper game.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
    
    def generate_factual_information(self, complexity: str = "medium") -> InformationSeed:
        """Generate factual information with specific data points."""
        
        if complexity == "simple":
            content = """The Eiffel Tower was built in 1889 in Paris, France. It stands 324 meters tall and was designed by Gustave Eiffel. The tower has 3 levels and receives about 7 million visitors each year."""
            key_elements = ["Eiffel Tower", "1889", "Paris", "France", "324 meters", "Gustave Eiffel", "3 levels", "7 million visitors"]
            
        elif complexity == "medium":
            content = """The Amazon rainforest covers approximately 5.5 million square kilometers across 9 countries, with Brazil containing 60% of the forest. It produces 20% of the world's oxygen and contains over 400 billion trees. The forest is home to 10% of known species, including 2.5 million insect species, 40,000 plant species, and 1,300 bird species. Deforestation rates peaked at 27,772 square kilometers in 2004 but have fluctuated since then."""
            key_elements = ["Amazon rainforest", "5.5 million square kilometers", "9 countries", "Brazil", "60%", "20% oxygen", "400 billion trees", "10% species", "2.5 million insects", "40,000 plants", "1,300 birds", "27,772 square kilometers", "2004"]
            
        else:  # complex
            content = """The Large Hadron Collider (LHC) at CERN, operational since 2008, is a 27-kilometer circular particle accelerator located 100 meters underground on the France-Switzerland border. It accelerates protons to 99.9999991% the speed of light, achieving collision energies of 13 TeV. The LHC consists of 1,232 dipole magnets cooled to -271.25°C using liquid helium. In 2012, it confirmed the existence of the Higgs boson, predicted by Peter Higgs in 1964. The facility employs 2,500 staff and costs approximately €4.75 billion to construct."""
            key_elements = ["Large Hadron Collider", "LHC", "CERN", "2008", "27-kilometer", "100 meters underground", "France-Switzerland", "99.9999991% speed of light", "13 TeV", "1,232 dipole magnets", "-271.25°C", "liquid helium", "2012", "Higgs boson", "Peter Higgs", "1964", "2,500 staff", "€4.75 billion"]
        
        return InformationSeed(
            content=content,
            information_type="factual",
            complexity_level=complexity,
            expected_key_elements=key_elements,
            target_length=len(content.split())
        )
    
    def generate_narrative_information(self, complexity: str = "medium") -> InformationSeed:
        """Generate narrative stories with characters and plot."""
        
        if complexity == "simple":
            content = """Sarah found an old key in her grandmother's attic. The key was made of brass and had strange symbols carved into it. She decided to search the house for what it might unlock. After checking every door and drawer, she discovered a small wooden box hidden behind books in the library. The key fit perfectly, and inside she found her grandmother's diary from 1943."""
            key_elements = ["Sarah", "old key", "grandmother's attic", "brass", "strange symbols", "wooden box", "library", "grandmother's diary", "1943"]
            
        elif complexity == "medium":
            content = """Detective Martinez arrived at the mansion at midnight, responding to reports of strange lights. The Victorian house had been empty for twenty years since the Blackwood family disappeared. As she approached the front door, she noticed fresh footprints in the mud leading to the garden. Following the trail, she discovered a hidden entrance to underground tunnels. Inside, she found evidence that someone had been living there recently: fresh food, modern electronics, and photographs of the missing family. The case that had puzzled the department for two decades suddenly had new leads."""
            key_elements = ["Detective Martinez", "mansion", "midnight", "strange lights", "Victorian house", "twenty years", "Blackwood family", "footprints", "garden", "underground tunnels", "fresh food", "modern electronics", "photographs", "two decades"]
            
        else:  # complex
            content = """Captain Elena Vasquez commanded the research vessel Aurora as it approached the Mariana Trench in the Pacific Ocean. Her team of marine biologists had been tracking unusual sonar readings for three months. Dr. James Chen, the expedition's lead scientist, reported that the signals originated from a depth of 11,000 meters, deeper than any known life forms. As they deployed the submersible Nereid, equipped with advanced cameras and sampling equipment, they encountered bioluminescent creatures unlike anything in scientific literature. The discovery would revolutionize understanding of deep-sea ecosystems, but first they had to survive the technical malfunction that left them stranded 2 kilometers below the surface with only 6 hours of oxygen remaining."""
            key_elements = ["Captain Elena Vasquez", "Aurora", "Mariana Trench", "Pacific Ocean", "marine biologists", "sonar readings", "three months", "Dr. James Chen", "11,000 meters", "submersible Nereid", "cameras", "sampling equipment", "bioluminescent creatures", "deep-sea ecosystems", "technical malfunction", "2 kilometers", "6 hours oxygen"]
        
        return InformationSeed(
            content=content,
            information_type="narrative",
            complexity_level=complexity,
            expected_key_elements=key_elements,
            target_length=len(content.split())
        )
    
    def generate_technical_information(self, complexity: str = "medium") -> InformationSeed:
        """Generate technical instructions or procedures."""
        
        if complexity == "simple":
            content = """To reset your router: 1) Unplug the power cable from the router. 2) Wait 30 seconds. 3) Plug the power cable back in. 4) Wait 2 minutes for the router to fully boot up. 5) Check that all indicator lights are green or blue. 6) Test your internet connection by opening a web browser."""
            key_elements = ["reset router", "unplug power cable", "30 seconds", "plug back in", "2 minutes", "indicator lights", "green or blue", "test internet", "web browser"]
            
        elif complexity == "medium":
            content = """To configure a secure SSH connection: 1) Generate an SSH key pair using 'ssh-keygen -t rsa -b 4096 -C "your_email@example.com"'. 2) Copy the public key to the server using 'ssh-copy-id username@server_ip'. 3) Edit the SSH daemon configuration file at /etc/ssh/sshd_config. 4) Set 'PasswordAuthentication no' and 'PubkeyAuthentication yes'. 5) Change the default port from 22 to a custom port like 2222. 6) Restart the SSH service with 'sudo systemctl restart sshd'. 7) Test the connection using 'ssh -p 2222 username@server_ip'."""
            key_elements = ["SSH connection", "ssh-keygen", "-t rsa -b 4096", "ssh-copy-id", "/etc/ssh/sshd_config", "PasswordAuthentication no", "PubkeyAuthentication yes", "port 22", "2222", "systemctl restart sshd", "ssh -p 2222"]
            
        else:  # complex
            content = """To implement a load-balanced Kubernetes cluster: 1) Initialize the master node with 'kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=<MASTER_IP>'. 2) Configure kubectl by copying admin.conf to ~/.kube/config with proper permissions (chmod 600). 3) Install a CNI plugin like Flannel using 'kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml'. 4) Join worker nodes using the token from step 1: 'kubeadm join <MASTER_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>'. 5) Deploy an NGINX ingress controller with 'kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/cloud/deploy.yaml'. 6) Configure horizontal pod autoscaling by creating HPA resources with CPU and memory thresholds at 70% and 80% respectively. 7) Set up persistent volumes using StorageClass definitions with provisioner 'kubernetes.io/aws-ebs' for AWS environments."""
            key_elements = ["Kubernetes cluster", "kubeadm init", "10.244.0.0/16", "apiserver-advertise-address", "kubectl", "admin.conf", "~/.kube/config", "chmod 600", "CNI plugin", "Flannel", "kubeadm join", "6443", "NGINX ingress", "controller-v1.0.0", "HPA", "70%", "80%", "StorageClass", "kubernetes.io/aws-ebs"]
        
        return InformationSeed(
            content=content,
            information_type="technical",
            complexity_level=complexity,
            expected_key_elements=key_elements,
            target_length=len(content.split())
        )
    
    def generate_structured_information(self, complexity: str = "medium") -> InformationSeed:
        """Generate structured data like lists or hierarchical information."""
        
        if complexity == "simple":
            content = """Shopping List: Fruits: apples (6), bananas (1 bunch), oranges (4). Vegetables: carrots (2 lbs), broccoli (1 head), spinach (1 bag). Dairy: milk (1 gallon), cheese (cheddar block), yogurt (6 cups). Meat: chicken breast (2 lbs), ground beef (1 lb)."""
            key_elements = ["Shopping List", "Fruits", "apples", "6", "bananas", "1 bunch", "oranges", "4", "Vegetables", "carrots", "2 lbs", "broccoli", "1 head", "spinach", "1 bag", "Dairy", "milk", "1 gallon", "cheese", "cheddar", "yogurt", "6 cups", "Meat", "chicken breast", "ground beef"]
            
        elif complexity == "medium":
            content = """Company Organization: Executive Level: CEO (Sarah Johnson), CFO (Michael Chen), CTO (Lisa Rodriguez). Department Heads: Marketing Director (Tom Wilson, 15 staff), Sales Director (Anna Kim, 22 staff), Engineering Manager (David Park, 18 developers), HR Director (Maria Garcia, 8 staff). Regional Offices: New York (125 employees), San Francisco (89 employees), London (67 employees), Tokyo (45 employees). Annual Budget: Marketing $2.5M, Sales $1.8M, Engineering $4.2M, Operations $3.1M, Total Revenue Target $45M."""
            key_elements = ["Company Organization", "CEO", "Sarah Johnson", "CFO", "Michael Chen", "CTO", "Lisa Rodriguez", "Marketing Director", "Tom Wilson", "15 staff", "Sales Director", "Anna Kim", "22 staff", "Engineering Manager", "David Park", "18 developers", "HR Director", "Maria Garcia", "8 staff", "New York", "125 employees", "San Francisco", "89 employees", "London", "67 employees", "Tokyo", "45 employees", "$2.5M", "$1.8M", "$4.2M", "$3.1M", "$45M"]
            
        else:  # complex
            content = """Global Climate Data Analysis: Temperature Trends (1990-2020): Arctic Region: +2.3°C average increase, ice coverage reduced by 13% annually. Tropical Regions: +1.1°C increase, precipitation patterns shifted 15% toward extreme events. Temperate Zones: +1.7°C increase, growing seasons extended by 18 days average. Ocean Data: Surface temperature +0.8°C, pH decreased by 0.1 units (30% increase in acidity), sea level rise 3.2mm annually. Carbon Emissions by Sector: Energy Production 25%, Transportation 14%, Industry 21%, Agriculture 24%, Buildings 6%, Other 10%. Renewable Energy Adoption: Solar capacity increased 890% (2010-2020), Wind capacity increased 340%, Hydroelectric remained stable at 16% of global capacity. Economic Impact: Climate adaptation costs estimated at $140-300 billion annually by 2030, renewable energy investments reached $303 billion in 2020."""
            key_elements = ["Climate Data", "1990-2020", "Arctic", "+2.3°C", "13% annually", "Tropical", "+1.1°C", "15% extreme events", "Temperate", "+1.7°C", "18 days", "Ocean", "+0.8°C", "pH 0.1", "30% acidity", "3.2mm annually", "Energy 25%", "Transportation 14%", "Industry 21%", "Agriculture 24%", "Buildings 6%", "Other 10%", "Solar 890%", "Wind 340%", "Hydroelectric 16%", "$140-300 billion", "2030", "$303 billion", "2020"]
        
        return InformationSeed(
            content=content,
            information_type="structured",
            complexity_level=complexity,
            expected_key_elements=key_elements,
            target_length=len(content.split())
        )
    
    def generate_information(self, info_type: str, complexity: str = "medium") -> InformationSeed:
        """Generate information of specified type and complexity."""
        generators = {
            "factual": self.generate_factual_information,
            "narrative": self.generate_narrative_information,
            "technical": self.generate_technical_information,
            "structured": self.generate_structured_information
        }
        
        if info_type not in generators:
            raise ValueError(f"Unknown information type: {info_type}. Available types: {list(generators.keys())}")
        
        return generators[info_type](complexity)
    
    def generate_batch(self, types: List[str], complexities: List[str], count_per_combination: int = 1) -> List[InformationSeed]:
        """Generate a batch of information seeds for systematic testing."""
        seeds = []
        for info_type in types:
            for complexity in complexities:
                for _ in range(count_per_combination):
                    seeds.append(self.generate_information(info_type, complexity))
        return seeds

"""
Country subset generation and management for analysis filtering.

This module generates JSON files containing lists of country IDs for different
geographic and economic groupings (continents, HDI levels, income groups, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_CONTINENTS_PATH = "data_nobackup/misc/raw/continents/continents.csv"
DEFAULT_COUNTRY_MAPPING_PATH = "data_nobackup/misc/processed/stage_2/gadm/country_code_mapping.json"
DEFAULT_OUTPUT_DIR = "data_nobackup/subsets"


class SubsetGenerator:
    """Generate country subset files for analysis filtering."""
    
    def __init__(
        self,
        project_root: Path,
        continents_path: Optional[str] = None,
        country_mapping_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize subset generator.
        
        Parameters:
        -----------
        project_root : Path
            Root directory of the project
        continents_path : str, optional
            Path to continents CSV file (relative to project_root)
        country_mapping_path : str, optional
            Path to country code mapping JSON (relative to project_root)
        output_dir : str, optional
            Output directory for subset files (relative to project_root)
        """
        self.project_root = Path(project_root)
        
        # Setup paths
        self.continents_path = self.project_root / (continents_path or DEFAULT_CONTINENTS_PATH)
        self.country_mapping_path = self.project_root / (country_mapping_path or DEFAULT_COUNTRY_MAPPING_PATH)
        self.output_dir = self.project_root / (output_dir or DEFAULT_OUTPUT_DIR)
        
        # Path to GADM geopackage for country name lookups
        self.gadm_path = self.project_root / "data_nobackup" / "misc" / "processed" / "stage_1" / "gadm" / "gadm_levelADM_0_simplified.gpkg"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.continents_df = None
        self.country_to_id = None
        self.gadm_name_to_code = None
        self._load_data()
    
    def _load_data(self):
        """Load continent and country mapping data."""
        if not self.continents_path.exists():
            logger.warning(f"Continents file not found: {self.continents_path}")
            return
        
        if not self.country_mapping_path.exists():
            logger.warning(f"Country mapping file not found: {self.country_mapping_path}")
            return
        
        try:
            # Load continent table
            self.continents_df = pd.read_csv(self.continents_path)
            logger.debug(f"Loaded continents data: {len(self.continents_df)} countries")
            
            # Load country ID mapping
            with open(self.country_mapping_path, 'r') as f:
                self.country_to_id = json.load(f)
            logger.debug(f"Loaded country mapping: {len(self.country_to_id)} countries")
            
            # Load GADM country name to code mapping if available
            if self.gadm_path.exists():
                import geopandas as gpd
                gadm = gpd.read_file(self.gadm_path).drop(columns=["geometry"])
                self.gadm_name_to_code = gadm.set_index("COUNTRY")["GID_0"].to_dict()
                logger.debug(f"Loaded GADM name mapping: {len(self.gadm_name_to_code)} countries")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def generate_continent_subsets(self) -> Dict[str, str]:
        """
        Generate subset files for each continent.
        
        Returns:
        --------
        dict mapping continent codes to output file paths
        """
        if self.continents_df is None or self.country_to_id is None:
            raise ValueError("Data not loaded. Cannot generate subsets.")
        
        output_files = {}
        
        # Filter out rows with missing continent codes
        valid_continents = self.continents_df.dropna(subset=['Continent_Code'])
        
        for continent_code in valid_continents['Continent_Code'].unique():
            # Skip if continent_code is not a string (safety check)
            if not isinstance(continent_code, str):
                logger.warning(f"Skipping invalid continent code: {continent_code}")
                continue
            
            # Get country codes for this continent
            country_codes = valid_continents.query(
                f"Continent_Code == '{continent_code}'"
            )['Three_Letter_Country_Code'].tolist()
            
            # Map to country IDs
            country_ids = [
                self.country_to_id[code]
                for code in country_codes
                if code in self.country_to_id
            ]
            
            # Skip if no countries mapped
            if not country_ids:
                logger.warning(f"No countries mapped for continent {continent_code}, skipping")
                continue
            
            # Get continent name for better filename
            continent_name = valid_continents.query(
                f"Continent_Code == '{continent_code}'"
            )['Continent_Name'].iloc[0] if not valid_continents.query(
                f"Continent_Code == '{continent_code}'"
            ).empty else continent_code
            
            # Save to JSON
            output_file = self.output_dir / f"continent_{continent_code.lower()}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'name': continent_name,
                    'code': continent_code,
                    'country_ids': country_ids,
                    'n_countries': len(country_ids)
                }, f, indent=2)
            
            output_files[continent_code] = str(output_file)
            logger.info(f"Generated subset for {continent_name} ({continent_code}): {len(country_ids)} countries")
        
        return output_files
    
    def generate_custom_subset(
        self,
        name: str,
        country_ids: Optional[List[int]] = None,
        country_codes: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Generate a custom subset file.
        
        Parameters:
        -----------
        name : str
            Name for the subset (used in filename)
        country_ids : list of int, optional
            List of country IDs to include
        country_codes : list of str, optional
            List of country codes to include (will be mapped to IDs)
        description : str, optional
            Description of the subset
        
        Returns:
        --------
        output_file : str
            Path to generated subset file
        """
        if country_ids is None and country_codes is None:
            raise ValueError("Must provide either country_ids or country_codes")
        
        if country_codes is not None:
            if self.country_to_id is None:
                raise ValueError("Country mapping not loaded")
            country_ids = [
                self.country_to_id[code]
                for code in country_codes
                if code in self.country_to_id
            ]
        
        # Save to JSON
        output_file = self.output_dir / f"custom_{name.lower().replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'name': name,
                'description': description or f"Custom subset: {name}",
                'country_ids': country_ids,
                'n_countries': len(country_ids)
            }, f, indent=2)
        
        logger.info(f"Generated custom subset '{name}': {len(country_ids)} countries")
        return str(output_file)
    
    def generate_hodler_raschky_2014_subset(self) -> str:
        """
        Generate subset for Hodler & Raschky (2014) countries.
        
        This includes the 136 countries used in:
        Hodler, R., & Raschky, P. A. (2014). Regional favoritism. 
        The Quarterly Journal of Economics, 129(2), 995-1033.
        
        Returns:
        --------
        output_file : str
            Path to generated subset file
        """
        if self.gadm_name_to_code is None or self.country_to_id is None:
            raise ValueError("GADM and country mapping data not loaded. Cannot generate subset.")
        
        # Hodler & Raschky 2014 countries
        hr2014_countries = [
            "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Australia",
            "Austria", "Bangladesh", "Belarus", "Belgium", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina",
            "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
            "Central African Republic", "Chad", "Chile", "China", "Colombia", "Costa Rica", "Czechia",
            "Côte d'Ivoire", "Democratic Republic of the Congo", "Denmark", "Timor-Leste", "Ecuador", "El Salvador",
            "Eritrea", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
            "Ghana", "Greece", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "India", "Indonesia",
            "Iran", "Iraq", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Laos", "Latvia", "Lebanon",
            "Liberia", "Lithuania", "North Macedonia", "Madagascar", "Malawi", "Malaysia", "Mali", "Mauritania", "México",
            "Mongolia", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand",
            "Nicaragua", "Niger", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan", "Panama",
            "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Republic of the Congo",
            "Russia", "Rwanda", "Senegal", "Serbia", "Sierra Leone", "Slovakia", "Slovenia", "Somalia", "South Africa",
            "South Korea", "Spain", "Sri Lanka", "Sudan", "Sweden", "Taiwan", "Tajikistan", "Tanzania", "Thailand",
            "Togo", "Tunisia", "Uganda", "Ukraine", "United Kingdom", "United States", "Uruguay", "Venezuela",
            "Vietnam", "Yemen", "Zambia", "Zimbabwe"
        ]
        
        # Map country names to IDs
        country_ids = []
        missing_countries = []
        
        for country_name in hr2014_countries:
            try:
                # Get ISO3 code from GADM name
                iso3_code = self.gadm_name_to_code.get(country_name)
                
                if iso3_code is None:
                    missing_countries.append(country_name)
                    continue
                
                # Get country ID from ISO3 code
                country_id = self.country_to_id.get(iso3_code)
                
                if country_id is None:
                    missing_countries.append(f"{country_name} (ISO3: {iso3_code})")
                    continue
                
                country_ids.append(country_id)
                
            except Exception as e:
                logger.warning(f"Failed to map country '{country_name}': {e}")
                missing_countries.append(country_name)
        
        if missing_countries:
            logger.warning(f"Could not map {len(missing_countries)} countries: {missing_countries[:5]}...")
        
        # Save to JSON
        output_file = self.output_dir / "research_hodler_raschky_2014.json"
        with open(output_file, 'w') as f:
            json.dump({
                'name': 'Hodler & Raschky (2014)',
                'description': 'Countries used in Hodler & Raschky (2014) Regional favoritism study',
                'reference': 'Hodler, R., & Raschky, P. A. (2014). Regional favoritism. The Quarterly Journal of Economics, 129(2), 995-1033.',
                'country_ids': sorted(country_ids),
                'n_countries': len(country_ids),
                'n_original': len(hr2014_countries),
                'n_missing': len(missing_countries)
            }, f, indent=2)
        
        logger.info(f"Generated Hodler & Raschky (2014) subset: {len(country_ids)}/{len(hr2014_countries)} countries mapped")
        
        if missing_countries:
            logger.info(f"Missing countries saved to: {output_file.parent / 'research_hodler_raschky_2014_missing.txt'}")
            with open(output_file.parent / 'research_hodler_raschky_2014_missing.txt', 'w') as f:
                f.write('\n'.join(missing_countries))
        
        return str(output_file)
    
    def generate_all_default_subsets(self) -> Dict[str, str]:
        """
        Generate all default subset files.
        
        Returns:
        --------
        dict mapping subset names to output file paths
        """
        output_files = {}
        
        # Generate continent subsets
        continent_files = self.generate_continent_subsets()
        output_files.update(continent_files)
        
        # Generate research subsets
        try:
            hr2014_file = self.generate_hodler_raschky_2014_subset()
            output_files['hodler_raschky_2014'] = hr2014_file
        except Exception as e:
            logger.warning(f"Failed to generate Hodler & Raschky (2014) subset: {e}")
        
        logger.info(f"Generated {len(output_files)} default subset files")
        return output_files
    
    @staticmethod
    def load_subset(subset_file: Path) -> List[int]:
        """
        Load country IDs from a subset file.
        
        Parameters:
        -----------
        subset_file : Path
            Path to subset JSON file
        
        Returns:
        --------
        list of int
            Country IDs in the subset
        """
        with open(subset_file, 'r') as f:
            data = json.load(f)
        return data['country_ids']


def generate_default_subsets(project_root: Path) -> None:
    """
    Generate all default subset files.
    
    Parameters:
    -----------
    project_root : Path
        Root directory of the project
    """
    generator = SubsetGenerator(project_root)
    output_files = generator.generate_all_default_subsets()
    
    print(f"\nGenerated {len(output_files)} subset files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    import sys
    
    # Get project root from command line or use default
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        # Assume script is in gnt/analysis/
        project_root = Path(__file__).parent.parent.parent
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate subsets
    generate_default_subsets(project_root)

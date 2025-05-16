import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

load_dotenv()

@dataclass
class EstimationParameters:
    # Base class for all estimation parameters
    pass

@dataclass
class LOCParameters(EstimationParameters):
    equivphyskloc: float = 1.0
    mode: str = "Organic"  # Organic, Semi-detached, or Embedded
    rely: float = 1.0
    data: float = 1.0
    cplx: float = 1.0
    time: float = 1.0
    stor: float = 1.0
    virt: float = 1.0
    turn: float = 1.0
    acap: float = 1.0
    aexp: float = 1.0
    pcap: float = 1.0
    vexp: float = 1.0
    lexp: float = 1.0
    modp: float = 1.0
    tool: float = 1.0
    sced: float = 1.0

@dataclass
class FPParameters(EstimationParameters):
    AFP: float = 0.0
    EI: int = 0  # External Inputs
    EO: int = 0  # External Outputs
    EQ: int = 0  # External Inquiries
    ELF: int = 0  # External Logical Files
    IFL: int = 0  # Internal Logical Files
    PDR_AFP: float = 0.0
    PDR_UFP: float = 0.0
    NPDR_AFP: float = 0.0
    NPDU_UFP: float = 0.0

@dataclass
class UCPParameters(EstimationParameters):
    # Actor counts
    simple_actors: int = 0
    average_actors: int = 0
    complex_actors: int = 0
    
    # Use case counts
    simple_uc: int = 0
    average_uc: int = 0
    complex_uc: int = 0
    
    # Complexity factors
    tcf: float = 1.0
    ecf: float = 1.0
    
    # Development environment
    language: str = "Java"
    methodology: str = "Waterfall" 
    application_type: str = "Business Application"
    
    # For serialization support
    def items(self):
        return {
            'Simple Actors': self.simple_actors,
            'Average Actors': self.average_actors,
            'Complex Actors': self.complex_actors,
            'Simple UC': self.simple_uc,
            'Average UC': self.average_uc,
            'Complex UC': self.complex_uc,
            'TCF': self.tcf,
            'ECF': self.ecf,
            'Language': self.language,
            'Methodology': self.methodology,
            'ApplicationType': self.application_type
        }
@dataclass
class ProjectAnalysis:
    suggested_method: str
    parameters: EstimationParameters
    explanation: str = ""
    raw_response: str = ""

class ProjectAnalyzer:
    def __init__(self, api_key: str = None):
        # Read API key from argument or environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        # Configure Gemini client
        genai.configure(api_key=self.api_key)
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_project(self, description: str) -> ProjectAnalysis:
        """Analyze project requirements and return structured analysis"""
        prompt = self._create_analysis_prompt(description)
        response_text = self._call_gemini_api(prompt)
        return self._parse_response(response_text, description)

    def _create_analysis_prompt(self, description: str) -> str:
        return f"""Analyze the following software project requirements and provide structured estimation guidance.
        
        Project Description:
        {description}
        
        I need you to:
        1. Recommend the most appropriate estimation method (LOC, FP, or UCP) for this project
        2. Extract or estimate concrete values for all parameters needed by the chosen method
        
        For LOC method, provide:
        - Estimated KLOC (thousand lines of code)
        - Development mode (Organic, Semi-detached, or Embedded)
        - All COCOMO cost drivers (rely, data, cplx, time, stor, virt, turn, acap, aexp, pcap, vexp, lexp, modp, tool, sced)
        
        For FP method, provide:
        - External Inputs (EI)
        - External Outputs (EO)
        - External Inquiries (EQ)
        - External Logic Files (ELF)
        - Internal Logic Files (ILF)
        - Productivity metrics if possible
        
        For UCP method, provide:
        - Actor counts (simple, average, complex)
        - Use case counts (simple, average, complex)
        - Technical Complexity Factor (TCF) estimate
        - Environmental Complexity Factor (ECF) estimate
        - Programming language, methodology, and application type
        
        Format your response in JSON as follows:
        {{
            "method": "LOC|FP|UCP",
            "parameters": {{
                // All relevant parameters for the selected method
            }},
            "explanation": "Brief explanation of why this method was chosen",
        }}
        
        Be as specific as possible with the parameter values. If you need to make an educated guess for any parameter, make your best estimate based on the project description. Focus on providing concrete, usable values.
        """

    def _call_gemini_api(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            
            # Debug log
            print("API Response:", response)
            
            # Check response format
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return "".join(part.text for part in response.parts)
            else:
                return f"Error: Unexpected response format: {type(response)}"
                
        except Exception as e:
            print(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        try:
            # Try to parse the entire text as JSON first
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex with improved pattern
            # Look for JSON patterns more aggressively
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                try:
                    json_text = json_match.group(1)
                    # Clean up potential truncated JSON
                    # Make sure we have matching braces
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    # Try a more lenient approach - extract method and parameters separately
                    method_match = re.search(r'"method"\s*:\s*"([^"]+)"', text)
                    if method_match:
                        method = method_match.group(1)
                        # Create a minimal valid JSON with just the method
                        return {
                            "method": method,
                            "parameters": {},
                            "explanation": "Partial data extracted from incomplete response"
                        }
                    return None
            return None
            
    def _parse_response(self, response_text: str, original_description: str) -> ProjectAnalysis:
        print("Raw response:", response_text)
        
        # Check if response is an error
        if isinstance(response_text, str) and response_text.startswith("Error:"):
            return ProjectAnalysis(
                suggested_method="Error",
                parameters=None,
                explanation=f"Failed to analyze: {response_text}",
                raw_response=response_text
            )
        
        # Try to extract JSON from the response
        data = self._extract_json(response_text)
        
        if not data:
            return ProjectAnalysis(
                suggested_method="Error",
                parameters=None,
                explanation="The AI response could not be parsed correctly. Please try again with a more detailed project description.",
                raw_response=response_text
            )
        
        # Get method and create appropriate parameters object
        method = data.get("method", "").upper()
        param_data = data.get("parameters", {})
        
        if method == "LOC":
            parameters = LOCParameters(
                equivphyskloc=float(param_data.get("equivphyskloc", param_data.get("KLOC", 1.0))),
                mode=param_data.get("mode", "Organic"),
                rely=float(param_data.get("rely", 1.0)),
                data=float(param_data.get("data", 1.0)),
                cplx=float(param_data.get("cplx", 1.0)),
                time=float(param_data.get("time", 1.0)),
                stor=float(param_data.get("stor", 1.0)),
                virt=float(param_data.get("virt", 1.0)),
                turn=float(param_data.get("turn", 1.0)),
                acap=float(param_data.get("acap", 1.0)),
                aexp=float(param_data.get("aexp", 1.0)),
                pcap=float(param_data.get("pcap", 1.0)),
                vexp=float(param_data.get("vexp", 1.0)),
                lexp=float(param_data.get("lexp", 1.0)),
                modp=float(param_data.get("modp", 1.0)),
                tool=float(param_data.get("tool", 1.0)),
                sced=float(param_data.get("sced", 1.0))
            )
        elif method == "FP":
            parameters = FPParameters(
                AFP=float(param_data.get("AFP", 0.0)),
                EI=int(param_data.get("EI", 0)),
                EO=int(param_data.get("EO", 0)),
                EQ=int(param_data.get("EQ", 0)),
                ELF=int(param_data.get("ELF", 0)),
                IFL=int(param_data.get("ILF", param_data.get("IFL", 0))),  # Handle both ILF and IFL
                PDR_AFP=float(param_data.get("PDR_AFP", 0.0)),
                PDR_UFP=float(param_data.get("PDR_UFP", 0.0)),
                NPDR_AFP=float(param_data.get("NPDR_AFP", 0.0)),
                NPDU_UFP=float(param_data.get("NPDU_UFP", 0.0))
            )
        elif method == "UCP":
            # Get actor counts
            actor_counts = param_data.get("actor_counts", {})
            use_case_counts = param_data.get("use_case_counts", {})
            
            # Calculate TCF and ECF if they're null
            tcf_factors = param_data.get("technical_complexity_factors", {})
            ecf_factors = param_data.get("environmental_complexity_factors", {})
            
            # Calculate TCF if null
            tcf = param_data.get("tcf")
            if tcf is None:
                # Default TCF calculation - ensure all values are numeric
                tcf_sum = 0
                for key, value in tcf_factors.items():
                    try:
                        tcf_sum += float(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        pass
                tcf = 0.6 + (0.01 * tcf_sum)
            else:
                # Ensure tcf is a float
                tcf = float(tcf)
            
            # Calculate ECF if null
            ecf = param_data.get("ecf")
            if ecf is None:
                # Default ECF calculation - ensure all values are numeric
                ecf_sum = 0
                for key, value in ecf_factors.items():
                    try:
                        ecf_sum += float(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        pass
                ecf = 1.4 + (-0.03 * ecf_sum)
            else:
                # Ensure ecf is a float
                ecf = float(ecf)
            
            # Safely convert actor and use case counts to integers
            simple_actors = actor_counts.get("simple", 0)
            average_actors = actor_counts.get("average", 0)
            complex_actors = actor_counts.get("complex", 0)
            simple_uc = use_case_counts.get("simple", 0)
            average_uc = use_case_counts.get("average", 0)
            complex_uc = use_case_counts.get("complex", 0)
            
            # Make sure all values are integers
            try:
                simple_actors = int(simple_actors)
            except (ValueError, TypeError):
                simple_actors = 0
                
            try:
                average_actors = int(average_actors)
            except (ValueError, TypeError):
                average_actors = 0
                
            try:
                complex_actors = int(complex_actors)
            except (ValueError, TypeError):
                complex_actors = 0
                
            try:
                simple_uc = int(simple_uc)
            except (ValueError, TypeError):
                simple_uc = 0
                
            try:
                average_uc = int(average_uc)
            except (ValueError, TypeError):
                average_uc = 0
                
            try:
                complex_uc = int(complex_uc)
            except (ValueError, TypeError):
                complex_uc = 0
            
            parameters = UCPParameters(
                simple_actors=simple_actors,
                average_actors=average_actors,
                complex_actors=complex_actors,
                simple_uc=simple_uc,
                average_uc=average_uc,
                complex_uc=complex_uc,
                tcf=tcf,
                ecf=ecf,
                language=param_data.get("programming_language", "Java"),
                methodology=param_data.get("methodology", "Waterfall"),
                application_type=param_data.get("application_type", "Business Application")
            )
        else:
            # Default to LOC if method is not recognized
            parameters = LOCParameters()
            method = "LOC"
        
        return ProjectAnalysis(
            suggested_method=method,
            parameters=parameters,
            explanation=data.get("explanation", "No explanation provided"),
            raw_response=response_text
        )

def add_ai_analysis_section(st):
    """Add AI analysis section to the Streamlit app"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Project Analysis")
    
    project_description = st.sidebar.text_area(
        "Project Description",
        height=150,
        help="Describe your project requirements for AI analysis"
    )
    
    if st.sidebar.button("Analyze with AI"):
        if not project_description.strip():
            st.sidebar.error("Please enter a project description for analysis")
            return None, None
            
        with st.spinner('Analyzing project requirements...'):
            try:
                analyzer = ProjectAnalyzer()
                analysis = analyzer.analyze_project(project_description)
                
                if analysis.suggested_method == "Error":
                    st.error(f"Analysis failed: {analysis.explanation}")
                    return None, None
                
                # Display results in expandable sections
                with st.expander("AI Analysis Results", expanded=True):
                    st.markdown(f"### Recommended Approach: {analysis.suggested_method}")
                    st.info(analysis.explanation)
                    
                    st.markdown("### Suggested Parameters")
                    params_dict = {}
                    
                    if analysis.suggested_method == "LOC":
                        loc_params = analysis.parameters
                        st.write(f"- KLOC: {loc_params.equivphyskloc}")
                        st.write(f"- Development Mode: {loc_params.mode}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Product Attributes:**")
                            st.write(f"- RELY: {loc_params.rely}")
                            st.write(f"- DATA: {loc_params.data}")
                            st.write(f"- CPLX: {loc_params.cplx}")
                            st.write(f"- TIME: {loc_params.time}")
                            st.write(f"- STOR: {loc_params.stor}")
                            st.write(f"- VIRT: {loc_params.virt}")
                            st.write(f"- TURN: {loc_params.turn}")
                        
                        with col2:
                            st.write("**Personnel Attributes:**")
                            st.write(f"- ACAP: {loc_params.acap}")
                            st.write(f"- AEXP: {loc_params.aexp}")
                            st.write(f"- PCAP: {loc_params.pcap}")
                            st.write(f"- VEXP: {loc_params.vexp}")
                            st.write(f"- LEXP: {loc_params.lexp}")
                            st.write(f"- MODP: {loc_params.modp}")
                            st.write(f"- TOOL: {loc_params.tool}")
                            st.write(f"- SCED: {loc_params.sced}")
                        
                        # Convert parameter values to dictionary format for return
                        params_dict = {
                            'equivphyskloc': loc_params.equivphyskloc,
                            'rely': loc_params.rely,
                            'data': loc_params.data,
                            'cplx': loc_params.cplx,
                            'time': loc_params.time,
                            'stor': loc_params.stor,
                            'virt': loc_params.virt,
                            'turn': loc_params.turn,
                            'acap': loc_params.acap,
                            'aexp': loc_params.aexp,
                            'pcap': loc_params.pcap,
                            'vexp': loc_params.vexp,
                            'lexp': loc_params.lexp,
                            'modp': loc_params.modp,
                            'tool': loc_params.tool,
                            'sced': loc_params.sced
                        }
                        
                        # Add mode information
                        mode = loc_params.mode
                        if mode == 'Organic':
                            params_dict['mode_embedded'] = 0
                            params_dict['mode_organic'] = 1
                            params_dict['mode_semidetached'] = 0
                        elif mode == 'Semi-detached':
                            params_dict['mode_embedded'] = 0
                            params_dict['mode_organic'] = 0
                            params_dict['mode_semidetached'] = 1
                        else:  # Embedded
                            params_dict['mode_embedded'] = 1
                            params_dict['mode_organic'] = 0
                            params_dict['mode_semidetached'] = 0
                        
                    elif analysis.suggested_method == "FP":
                        fp_params = analysis.parameters
                        st.write(f"- AFP: {fp_params.AFP}")
                        st.write(f"- External Inputs: {fp_params.EI}")
                        st.write(f"- External Outputs: {fp_params.EO}")
                        st.write(f"- External Inquiries: {fp_params.EQ}")
                        st.write(f"- External Logic Files: {fp_params.ELF}")
                        st.write(f"- Internal Logic Files: {fp_params.IFL}")
                        
                        if fp_params.PDR_AFP > 0:
                            st.write(f"- PDR_AFP: {fp_params.PDR_AFP}")
                        if fp_params.PDR_UFP > 0:
                            st.write(f"- PDR_UFP: {fp_params.PDR_UFP}")
                        if fp_params.NPDR_AFP > 0:
                            st.write(f"- NPDR_AFP: {fp_params.NPDR_AFP}")
                        if fp_params.NPDU_UFP > 0:
                            st.write(f"- NPDU_UFP: {fp_params.NPDU_UFP}")
                        
                        # Convert parameter values to dictionary format for return
                        params_dict = {
                            'AFP': fp_params.AFP,
                            'Input': fp_params.EI,
                            'Output': fp_params.EO,
                            'Enquiry': fp_params.EQ,
                            'File': fp_params.ELF,
                            'Interface': fp_params.IFL,
                            'PDR_AFP': fp_params.PDR_AFP,
                            'PDR_UFP': fp_params.PDR_UFP,
                            'NPDR_AFP': fp_params.NPDR_AFP,
                            'NPDU_UFP': fp_params.NPDU_UFP
                        }
                        
                    elif analysis.suggested_method == "UCP":
                        ucp_params = analysis.parameters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Actors:**")
                            st.write(f"- Simple Actors: {ucp_params.simple_actors}")
                            st.write(f"- Average Actors: {ucp_params.average_actors}")
                            st.write(f"- Complex Actors: {ucp_params.complex_actors}")
                        
                        with col2:
                            st.write("**Use Cases:**")
                            st.write(f"- Simple Use Cases: {ucp_params.simple_uc}")
                            st.write(f"- Average Use Cases: {ucp_params.average_uc}")
                            st.write(f"- Complex Use Cases: {ucp_params.complex_uc}")
                        
                        st.write("**Complexity Factors:**")
                        st.write(f"- Technical Complexity Factor (TCF): {ucp_params.tcf}")
                        st.write(f"- Environmental Complexity Factor (ECF): {ucp_params.ecf}")
                        
                        st.write("**Development Environment:**")
                        st.write(f"- Programming Language: {ucp_params.language}")
                        st.write(f"- Methodology: {ucp_params.methodology}")
                        st.write(f"- Application Type: {ucp_params.application_type}")
                        
                        # Calculate UAW and UUCW for display
                        uaw = (ucp_params.simple_actors * 1 + 
                               ucp_params.average_actors * 2 + 
                               ucp_params.complex_actors * 3)
                        
                        uucw = (ucp_params.simple_uc * 5 +
                                ucp_params.average_uc * 10 +
                                ucp_params.complex_uc * 15)
                        
                        st.write(f"- Calculated UAW: {uaw}")
                        st.write(f"- Calculated UUCW: {uucw}")
                        st.write(f"- Calculated UCP: {(uaw + uucw) * ucp_params.tcf * ucp_params.ecf}")
                        
                        # Convert parameter values to dictionary format for return
                        params_dict = {
                            'Simple Actors': ucp_params.simple_actors,
                            'Average Actors': ucp_params.average_actors,
                            'Complex Actors': ucp_params.complex_actors,
                            'Simple UC': ucp_params.simple_uc,
                            'Average UC': ucp_params.average_uc,
                            'Complex UC': ucp_params.complex_uc,
                            'TCF': ucp_params.tcf,
                            'ECF': ucp_params.ecf,
                            'Language': ucp_params.language,
                            'Methodology': ucp_params.methodology,
                            'ApplicationType': ucp_params.application_type,
                            'UAW': uaw,
                            'UUCW': uucw,
                            'UCP': (uaw + uucw) * ucp_params.tcf * ucp_params.ecf,
                            'Real_P20': 20  # default productivity factor
                        }
                    
                    
                    if st.button("Apply Suggested Parameters"):
                        return analysis.suggested_method, params_dict
                
                # Return data even if user doesn't click "Apply"
                return analysis.suggested_method, params_dict
                
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return None, None
    
    return None, None
# The 3D Maxwell Project

This code was developed by Pascal Kraft in an effort to generate a fast code to compute optimal shapes of 3D-waveguides based on a solution of the full Maxwell-Problem without simplifications based on non-physical assumptions. 

Core elements are a refined adjoint based optimization scheme and a sweeping preconditioner. 

This is a work in progress and not currently ready for general usage. Please refer to pascal.kraft@kit.edu for more information.

There are currently works in the branch complex-numbers that might need to be pulle.

# Naming convenctions

| Expression |Meaning     |
|------------|------------------------------------------------------------------------|
| Sector     | A sector is an expression used in shape modeling and refers to a subset of the area between the connectors. |
| Layer      | used in the implementation and represents the part of the triangulation owned by one Process.  |
| Connector  | The structure in the two halfspaces towards z->\infty and z->-\infty    |
| Space Transformation | The space transformation always also includes the computation of PML or other ABCs. it does not however involve the multiplication with the actual material property. This can be done at the point of usage. |
| ... | ... |

# Thanks

My thanks go to the CRC 1173 which is funding my research.

# Test Results

<h1>Test Report</h1>
        <div class="testsuite">
            <h2>Test Suite: DOFManagerTests</h2><a name="ca61df20-7641-4823-a74d-86f319a649c9">
            
            
            <table>
            <tr><th align="left">Duration</th><td align="right">0.0 sec</td></tr>
            <tr><th align="left">Test Cases</th><td align="right">1</td></tr>
            <tr><th align="left">Failures</th><td align="right">0</td></tr>
            
            
            </table>
            <a name="toc"></a>
            <h2>Results Index</h2>
            
        <ul>
            
            
            <li>All Test Classes
            <ul>
            <li>
                <a href="#6800a789-55dd-4607-ac6b-b930df62755f">DOFManagerTests</a>
                <ul>
                
                    <li>
                        <a href="#8d1cb15e-96f1-4c81-9abe-060a3ad2ce87">StaticFunctionTest</a>
                    </li>
                    
                </ul>
            </li>
            </ul>
            </li>
        </ul>
        
            <hr size="2"/>
            <h2>Test Results</h2>
            <div class="testclasses">
            
        <hr size="2"/>
        <a name="6800a789-55dd-4607-ac6b-b930df62755f">
        <div class="testclass">
            <div>Test Class: DOFManagerTests</div>
            <div class="testcases">
            
    <a name="8d1cb15e-96f1-4c81-9abe-060a3ad2ce87">
        <div class="testcase">
            <div class="details">
                <span class="testname"><b>StaticFunctionTest</b></span><br/>
                <span class="testclassname">DOFManagerTests</span><br/>
                <span class="duration">Time Taken: 0.0s</span>
            </div>
            
            
            <hr size="1"/>
            
            
        </div>
    </a>
        
            </div>
        </div>
        </a>
        
            </div>
        </div>

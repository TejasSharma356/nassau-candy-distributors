# Nassau Candy Distributor: Supply Chain Resilience & Route Efficiency
**Executive Summary for Government & Strategic Stakeholders**

## 1. Objective and Strategic Importance
Nassau Candy Distributor operates one of the most extensive confectionery wholesale networks in the United States. Maintaining a highly reliable and efficient supply chain is critical not only for corporate profitability but also for regional economic stability, ensuring consistent stock levels for thousands of retail partners.

The objective of this project was to audit historical shipping performance, identify structural delivery bottlenecks, and construct an **AI-driven logistics monitoring system**. This system empowers decision-makers with live analytics and predictive insights to preemptively mitigate supply chain disruptions.

## 2. Key Findings & Vulnerabilities
An exhaustive Exploratory Data Analysis (EDA) of historical logistics data revealed several systemic inefficiencies:
* **The "Coastal Squeeze"**: Delivery routes to the West Coast (e.g., California, Washington) and specific East Coast corridors experience disproportionately high delay rates irrespective of the dispatching factory.
* **Shipment Mode Inefficiencies**: While "Same Day" and "First Class" shipments generally adhere to SLAs, "Standard Class" cross-country logistics exhibit extreme variance, suggesting an over-reliance on vulnerable ground-freight corridors.
* **Cost-Time Imbalance**: Certain long-haul routes demonstrate diminishing returns, where expedited shipping costs spike exponentially without a proportional guarantee of on-time delivery.

## 3. The AI-Driven Solution
To address these vulnerabilities, we developed and deployed a **Live Route Efficiency Dashboard** backed by a custom Machine Learning engine.
* **Live Monitoring**: The system dynamically grades shipping routes on an "Efficiency Score" (balancing speed vs. cost) and maps geographic risk zones.
* **Predictive Delay Modeling**: A Random Forest Machine Learning classifier was trained on historical anomalies. When a dispatch planner inputs an order's parameters (destination, factory, units, and mode), the AI calculates the exact probability of a delay.
* **Strategic Tuning (Recall > Precision)**: The AI is deliberately tuned to prioritize *Recall* (83%). In supply chain risk management, a false alarm is exponentially cheaper than an unforeseen disruption. The model successfully flags 8 out of 10 systemic delays before the truck leaves the dock.

## 4. Economic & Operational Impact
By deploying this predictive infrastructure, management and regulatory stakeholders can expect:
1. **Targeted Infrastructure Investment**: Analytics pinpoint exactly which transit corridors require renegotiated vendor contracts or alternative routing.
2. **Proactive Customer Communication**: Retailers can be notified of predicted delays days in advance, absorbing the economic shock of stockouts.
3. **Agile Factory Allocation**: Dispatchers can dynamically reroute high-risk orders to closer facilities, reducing carbon footprint, lowering freight costs, and securing the supply chain.

---
*Prepared by Tejas — Data Analytics Intern, Unified Mentor*

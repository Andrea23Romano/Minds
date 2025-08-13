from minds.kernel.kernel import Kernel
from minds.agents.system import InterfaceAgent, INTERFACE_AGENT_ID
from minds.agents.worker import WorkerAgent

def test_interface_agent_initialization():
    """
    Tests that the InterfaceAgent is initialized with its well-known ID.
    """
    # The agent requires a kernel instance, even if it's not used in this test
    kernel = Kernel()
    agent = InterfaceAgent(kernel)
    assert agent.agent_id == INTERFACE_AGENT_ID

def test_worker_agent_initialization(mocker):
    """
    Tests that the WorkerAgent can be initialized.
    Mocks environment variables to avoid real API key dependency.
    """
    # Mock os.getenv to simulate the API key being present
    mocker.patch('os.getenv', return_value='fake_api_key_for_testing')

    kernel = Kernel()
    agent = WorkerAgent(kernel)
    assert agent is not None
    assert agent.agent_id is not None # It should have a generated UUID
    assert agent.agent_id != INTERFACE_AGENT_ID

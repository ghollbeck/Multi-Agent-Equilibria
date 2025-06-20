#!/usr/bin/env python3
"""
Focused test for memory system functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.')

def test_memory_system():
    """Test basic memory system functionality"""
    print("Testing MemoryManager with enable_shared_memory parameter...")
    
    try:
        from memory_storage import MemoryManager, AgentMemory
        
        mm1 = MemoryManager(retention_rounds=3, enable_shared_memory=False)
        print('✓ MemoryManager created without shared memory')
        
        mm2 = MemoryManager(retention_rounds=3, enable_shared_memory=True)
        print('✓ MemoryManager created with shared memory')
        
        agent_mem = mm1.create_agent_memory('TestAgent')
        print('✓ Agent memory created successfully')
        
        context = agent_mem.get_memory_context_for_decision()
        print('✓ Memory context methods work')
        print(f'  Context: {context[:50]}...')
        
        agent_mem.add_decision_memory(
            round_number=1,
            order_quantity=10,
            inventory=100,
            backlog=5,
            reasoning="Test reasoning",
            confidence=0.8
        )
        
        updated_context = agent_mem.get_memory_context_for_decision()
        print('✓ Memory addition and retrieval works')
        
        shared_context = mm2.get_shared_memory_context()
        print('✓ Shared memory context retrieval works')
        print(f'  Shared context: {shared_context[:50]}...')
        
        print('\n🎉 All memory system tests passed!')
        return True
        
    except Exception as e:
        print(f'✗ Memory system test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_system()
    sys.exit(0 if success else 1)

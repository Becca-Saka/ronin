import 'package:flutter/material.dart';

class E2eCounterPageNo extends StatefulWidget {
  const E2eCounterPageNo({super.key});

  @override
  State<E2eCounterPageNo> createState() => _E2eCounterPageNoState();
}

class _E2eCounterPageNoState extends State<E2eCounterPageNo> {
  int _counter = 0;

  void _increment() {
    setState(() => _counter++);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('E2E Test App'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Semantics(
              label: 'e2e_counter_prompt',
              child: const Text('Tap increment to increase the counter'),
            ),
            const SizedBox(height: 16),
            Semantics(
              label: 'counter_value',
              value: '$_counter',
              child: ExcludeSemantics(
                child: Text(
                  '$_counter',
                  style: Theme.of(context).textTheme.displayMedium,
                ),
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: Semantics(
        label: 'increment_button',
        button: true,
        child: FloatingActionButton(
          onPressed: _increment,
          tooltip: 'Increment',
          child: const Icon(Icons.add),
        ),
      ),
    );
  }
}

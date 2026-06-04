import 'package:flutter/material.dart';

class E2eCounterPage extends StatefulWidget {
  const E2eCounterPage({super.key});

  @override
  State<E2eCounterPage> createState() => _E2eCounterPageState();
}

class _E2eCounterPageState extends State<E2eCounterPage> {
  int _counter = 0;

  void _increment() {
    setState(() => _counter++);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Semantics(
          label: 'e2e_app_title',
          header: true,
          child: const Text('E2E Test App'),
        ),
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
